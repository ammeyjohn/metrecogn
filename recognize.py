from sys import argv
from os import path, listdir, mkdir
from shutil import copyfile, rmtree
import cv2
import numpy as np
import mxnet as mx
import utils.rotate as rot
import matplotlib.pyplot as plt

from collections import namedtuple

DEBUG = False

Batch = namedtuple('Batch', ['data'])

letters = list('ABCDEFGHIJKLMNOPQQRSTUVWXYZ')
cls_labels = ['LX14', 'LX15', 'WS', 'blurred', 'multi_meters', 'nometer', 'others', 'small_meter']
rot_labels = ['0', '180', 'L90', 'R90']


def get_module(prefix, num_epoch, img_shape=(224, 224)):
    print(prefix)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, num_epoch)
    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, img_shape[0], img_shape[1]))])
    mod.set_params(arg_params, aux_params, allow_missing=True)
    return mod


def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img


def classify(mod, img, labels):
    img = preprocess(img)
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    return prob[a[0]], labels[a[0]]


def classify_batch(prefix, num_epoch, labels):

    for l in labels:
        dst = path.join(arg_output, l)
        if path.exists(dst):
            rmtree(dst)
        mkdir(dst)
        if not path.exists(dst):
            mkdir(dst)

    mod = get_module(prefix, num_epoch)

    for file in listdir(arg_folder):

        img = cv2.imread(path.join(arg_folder, file))
        if img is None: continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        prob, label = classify(mod, img, labels)
        print('%s, prob=%f, label=%s' % (file, prob, label))

        src = path.join(arg_folder, file)
        dst = path.join(arg_output, label, file)

        copyfile(src, dst)


def get_regions(img, scale, min_neighbors):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # if DEBUG:
    #     plt.imshow(gray, cmap='gray')
    #     plt.show()

    classifier = cv2.CascadeClassifier('./output/cascade/cascade.xml')
    rects = classifier.detectMultiScale(gray, scale, minNeighbors=min_neighbors)
    regions = []
    for (x, y, w, h) in rects:
        rgn_img = img[y:y + h, x:x + w]

        if DEBUG:
            plt.imshow(rgn_img)
            plt.show()

        regions.append(rgn_img)
    return regions


def normalize(img):
    # 区分是否为水表
    cls_prob, cls_label = classify(cls_mod, img, cls_labels)
    if cls_label == 'LX14' or cls_label == 'LX15' or cls_label == 'WS' or cls_label == 'small_meter' or cls_label == 'blurred':

        # 获取图片旋转角度
        rot_prob, rot_label = classify(rot_mod, img, rot_labels)

        print('Classify Label %s, prob=%f; Rotation Label %s, prob=%f' % (cls_label, cls_prob, rot_label, rot_prob))

        if rot_label == '0':
            rot_img = img
        elif rot_label == '180':
            rot_img = rot.rotate180(img)
        elif rot_label == 'L90':
            rot_img = rot.rotate90(img)
        elif rot_label == 'R90':
            rot_img = rot.rotate270(img)

        # if DEBUG:
        #     plt.imshow(rot_img)
        #     plt.show()

        info = {
            'cls_label': cls_label,
            'cls_prob': str(cls_prob),
            'rot_label': rot_label,
            'rot_prob': str(rot_prob)
        }

        return rot_img, info

    return None, None


def cut_batch():
    count = 0
    for file in listdir(arg_folder):
        if file == '.DS_Store':
            continue

        img = cv2.imread(path.join(arg_folder, file))
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 获取水表表盘区域
        nor_img, info = normalize(img)
        if nor_img is None:
            continue

        # 截取读数区域
        regions = get_regions(nor_img, 1.1, 5)
        name, ext = path.splitext(file)
        for idx, reg_img in enumerate(regions):
            num = name[:-1]
            for l in letters:
                new_file = num + l + ext
                if not path.exists(path.join(arg_output, new_file)):
                    reg_img = cv2.cvtColor(reg_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(path.join(arg_output, new_file), reg_img)
                    break


def get_network():
    data = mx.symbol.Variable('data')
    conv1 = mx.symbol.Convolution(data=data, kernel=(5, 5), num_filter=32)
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(2, 2), stride=(1, 1))
    relu1 = mx.symbol.Activation(data=pool1, act_type="relu")

    conv2 = mx.symbol.Convolution(data=relu1, kernel=(5, 5), num_filter=32)
    pool2 = mx.symbol.Pooling(data=conv2, pool_type="avg", kernel=(2, 2), stride=(1, 1))
    relu2 = mx.symbol.Activation(data=pool2, act_type="relu")

    conv3 = mx.symbol.Convolution(data=relu2, kernel=(3, 3), num_filter=32)
    pool3 = mx.symbol.Pooling(data=conv3, pool_type="avg", kernel=(2, 2), stride=(1, 1))
    relu3 = mx.symbol.Activation(data=pool3, act_type="relu")

    conv4 = mx.symbol.Convolution(data=relu3, kernel=(3, 3), num_filter=32)
    pool4 = mx.symbol.Pooling(data=conv4, pool_type="avg", kernel=(2, 2), stride=(1, 1))
    relu4 = mx.symbol.Activation(data=pool4, act_type="relu")

    flatten = mx.symbol.Flatten(data=relu4)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=320)
    fc21 = mx.symbol.FullyConnected(data=fc1, num_hidden=10)
    fc22 = mx.symbol.FullyConnected(data=fc1, num_hidden=10)
    fc23 = mx.symbol.FullyConnected(data=fc1, num_hidden=10)
    fc24 = mx.symbol.FullyConnected(data=fc1, num_hidden=10)
    fc25 = mx.symbol.FullyConnected(data=fc1, num_hidden=10)
    fc2 = mx.symbol.Concat(*[fc21, fc22, fc23, fc24, fc25], dim=0)
    return mx.symbol.SoftmaxOutput(data=fc2, name="softmax")


def get_executor():
    batch_size = 1
    _, arg_params, aux_params = mx.model.load_checkpoint("./output/recognize/cnn-ocr", 1)
    data_shape = [("data", (batch_size, 3, 60, 150))]
    input_shapes = dict(data_shape)
    sym = get_network()
    executor = sym.simple_bind(ctx=mx.cpu(), **input_shapes)
    for key in executor.arg_dict.keys():
        if key in arg_params:
            arg_params[key].copyto(executor.arg_dict[key])
    return executor


def predict(img):
    img = cv2.resize(img, (150, 60))
    img = np.multiply(img, 1 / 255.0)
    img = img.transpose(2, 0, 1)

    executor.forward(is_train=False, data=mx.nd.array([img]))
    probs = executor.outputs[0].asnumpy()
    line = ''
    for i in range(probs.shape[0]):
        line += str(np.argmax(probs[i]))
    return line


def recognize(filepath):
    if not path.exists(filepath):
        return None

    img = cv2.imread(filepath)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 获取水表表盘区域
    nor_img, info = normalize(img)

    # 截取读数区域
    regions = get_regions(nor_img, 1.1, 5)

    # 识别表盘数字
    numbers = []
    for reg_img in regions:
        num = predict(reg_img)
        if num != '00000':
            numbers.append(num)

    return numbers, info


cls_mod = get_module('./output/classify/meter', 5)
rot_mod = get_module('./output/rotate/rotate', 5)
executor = get_executor()

if __name__ == '__main__':
    # arg_prefix = 'output/rotate/rotate'
    # classify_batch(arg_prefix, rotate_labels)
    # cut()
    #
    # folder = argv[1]
    # output = argv[2]

    arg_folder = argv[1]
    arg_output = argv[2]
    # cut_batch()
    classify_batch('./output/classify/meter', 5, cls_labels)

    # numbers = recognize(path.join(arg_folder, arg_name))
    # print(numbers)
