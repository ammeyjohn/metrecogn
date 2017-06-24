from sys import argv
import mxnet as mx

import logging

head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)


def get_iterators(batch_size, data_shape=(3, 224, 224)):
    train = mx.io.ImageRecordIter(
        path_imgrec='./output/classify/meter_train.rec',
        # path_imgrec='./output/rotate/rotate_train.rec',
        data_name='data',
        label_name='softmax_label',
        batch_size=batch_size,
        data_shape=data_shape,
        shuffle=True,
        rand_crop=True,
        rand_mirror=True)
    val = mx.io.ImageRecordIter(
        path_imgrec='./output/classify/meter_val.rec',
        # path_imgrec='./output/rotate/rotate_val.rec',
        data_name='data',
        label_name='softmax_label',
        batch_size=batch_size,
        data_shape=data_shape,
        rand_crop=False,
        rand_mirror=False)
    return train, val


# Train


def get_fine_tune_model(symbol, arg_params, num_classes, layer_name='flatten0'):
    """
    symbol: the pretrained network symbol
    arg_params: the argument parameters of the pretrained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = symbol.get_internals()
    net = all_layers[layer_name + '_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k: arg_params[k] for k in arg_params if 'fc1' not in k})
    return net, new_args


def fit(symbol, arg_params, aux_params, train, val, epoch, batch_size, num_gpus):
    devs = [mx.gpu(i) for i in range(num_gpus)]
    mod = mx.mod.Module(symbol=symbol, context=devs)
    mod.fit(train, val,
            num_epoch=epoch,
            arg_params=arg_params,
            aux_params=aux_params,
            allow_missing=True,
            batch_end_callback=mx.callback.Speedometer(batch_size, 100),
            kvstore='device',
            optimizer='sgd',
            optimizer_params={'learning_rate': 0.01},
            initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
            eval_metric='acc')
    metric = mx.metric.Accuracy()
    return mod, mod.score(val, metric)


if __name__ == '__main__':

    num_epoch = int(argv[1])

    num_classes = 8
    batch_per_gpu = 8
    num_gpus = 1

    sym, arg_params, aux_params = mx.model.load_checkpoint('./output/imagenet/Inception', 9)
    # mx.viz.plot_network(sym).view()

    (new_sym, new_args) = get_fine_tune_model(sym, arg_params, num_classes, 'flatten')

    batch_size = batch_per_gpu * num_gpus
    (train, val) = get_iterators(batch_size)
    mod, mod_score = fit(new_sym, new_args, aux_params, train, val, num_epoch, batch_size, num_gpus)
    print(mod_score)
    # assert mod_score > 0.77, "Low training accuracy."

    mod.save_checkpoint('./output/classify/meter', num_epoch)
    # mod.save_checkpoint('./output/rotate/rotate', num_epoch)
