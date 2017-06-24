# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
from sys import argv
import mxnet as mx
import numpy as np
from gen_number import *
from matplotlib import pyplot as plt
from collections import namedtuple
from cnn_ocr import *

def get_ocrnet():
    data = mx.symbol.Variable('data')
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=32)
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(2,2), stride=(1, 1))
    relu1 = mx.symbol.Activation(data=pool1, act_type="relu")

    conv2 = mx.symbol.Convolution(data=relu1, kernel=(5,5), num_filter=32)
    pool2 = mx.symbol.Pooling(data=conv2, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu2 = mx.symbol.Activation(data=pool2, act_type="relu")

    conv3 = mx.symbol.Convolution(data=relu2, kernel=(3,3), num_filter=32)
    pool3 = mx.symbol.Pooling(data=conv3, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu3 = mx.symbol.Activation(data=pool3, act_type="relu")

    conv4 = mx.symbol.Convolution(data=relu3, kernel=(3,3), num_filter=32)
    pool4 = mx.symbol.Pooling(data=conv4, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu4 = mx.symbol.Activation(data=pool4, act_type="relu")

    flatten = mx.symbol.Flatten(data = relu4)
    fc1 = mx.symbol.FullyConnected(data = flatten, num_hidden = 256)
    fc21 = mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
    fc22 = mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
    fc23 = mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
    fc24 = mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
    fc25 = mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
    fc2 = mx.symbol.Concat(*[fc21, fc22, fc23, fc24, fc25], dim = 0)
    return mx.symbol.SoftmaxOutput(data = fc2, name = "softmax")

def init():

    batch_size = 1
    _, arg_params, aux_params = mx.model.load_checkpoint("output/cnn-ocr", 1)
    data_shape = [("data", (batch_size, 3, 60, 150))]
    input_shapes = dict(data_shape)
    sym = get_ocrnet()
    executor = sym.simple_bind(ctx = mx.cpu(), **input_shapes)
    for key in executor.arg_dict.keys():
        if key in arg_params:
            arg_params[key].copyto(executor.arg_dict[key])

    return executor

def predict(executor, img):
    # Batch = namedtuple('Batch', ['data'])

    executor.forward(is_train = False, data = mx.nd.array([img]))
    probs = executor.outputs[0].asnumpy()

    line = ''
    for i in range(probs.shape[0]):
        line += str(np.argmax(probs[i]))
    return line



if __name__ == '__main__':

    # num, img = gen_sample(150, 60)
    # print('gen captcha:', num)
    # line = predict(img)
    # print('predicted: ' + line)

    folder = argv[1]
    filename = argv[2]
    # folder = 'plate'
    # filename = 'A00017.png'

    num = filename[1:-4]
    img = cv2.imread(os.path.join(folder, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
    img = cv2.resize(img, (150, 60))
    img = np.multiply(img, 1 / 255.0)
    img = img.transpose(2, 0, 1)

    exec = init()

    print('gen captcha:', num)
    line = predict(exec, [img])
    print('predicted: ' + line)

    # nums, images = [], []
    # for i, file in enumerate(os.listdir('plate')):
    #     if file[0] == '.':
    #         continue
    #     if i == 5:
    #         break
    #
    #     num = file[1:-4]
    #     nums.append(num)
    #
    #     img = cv2.imread(os.path.join('plate', file))
    #     # full = os.path.join('plate', 'B00195.png')
    #     # img = cv2.imread(full)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = cv2.resize(img, (150, 60))
    #     img = np.multiply(img, 1 / 255.0)
    #     img = img.transpose(2, 0, 1)
    #
    #     images.append(img)
    #
    # preds = predict(np.ndarray(images))
    #
    # count, hitted = len(nums), 0
    # for i in range(len(nums)-1):
    #     print('num=%s, predict=%s' % (nums[i], preds[i]))
    #
    #     if num[i] == preds[i]:
    #         hitted += 1
    #
    # print('count=%d, hitted=%d, percent=%f' % (count, hitted, 1.0 * hitted / count))
