import mxnet as mx
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
from os import listdir, path
# define a simple data batch
from collections import namedtuple

# sym, arg_params, aux_params = mx.model.load_checkpoint('./output/classify/meter', 5)
sym, arg_params, aux_params = mx.model.load_checkpoint('./output/rotate/rotate', 5)
mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
mod.set_params(arg_params, aux_params, allow_missing=True)
# with open('./output/imagenet/synset.txt', 'r') as f:
#     labels = [l.rstrip() for l in f]
labels = ['0', '180', 'L90', 'R90']

Batch = namedtuple('Batch', ['data'])


def get_image(url, show=False):
    # download and show the image
    # fname = mx.test_utils.download(url)
    img = cv2.cvtColor(cv2.imread(url), cv2.COLOR_BGR2RGB)
    if img is None:
        return None
    if show:
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img


def predict(url):
    img = get_image(url, show=True)

    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    # for i in a[0:5]:
    #     print('probability=%f, class=%s' % (prob[i], labels[i]))
    print('probability=%f, class=%s' % (prob[a[0]], labels[a[0]]))


if __name__ == '__main__':
    # predict('./images/classified/LX15/5e75240dc6c8d0b66f90dc08562f9949.jpg')

    folder = './images/original/random'
    files = list(listdir(folder))
    idx = random.randint(0, len(files)-1)
    predict(path.join(folder, files[idx]))
