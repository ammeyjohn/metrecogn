from sys import argv
from os import listdir, path
import math
import cv2

try:
    import rename as re
except:
    import utils.rename as re


def rotate(img, angle):

    cols, rows, _ = img.shape
    scale = 1
    if angle % 90 == 0:
        scale = float(max(cols, rows)) / min(cols, rows)
    else:
        scale = math.sqrt(pow(cols, 2) + pow(rows, 2)) / min(cols, rows)

    M = cv2.getRotationMatrix2D((rows/2, cols/2), angle, scale)
    res = cv2.warpAffine(img, M, (cols, rows))
    return res

def rotate90(img):
    F_img = cv2.flip(img, -1)
    T_img = cv2.transpose(F_img)
    return cv2.flip(T_img, 0)

def rotate180(img):
    return cv2.flip(img, -1)


def rotate270(img):
    T_img = cv2.transpose(img)
    return cv2.flip(T_img, 0)


if __name__ == '__main__':

    folder = argv[1]
    output = argv[2]

    for file in listdir(folder):
        img = cv2.imread(path.join(folder, file))
        if img is None:
            continue
        rot_img = rotate180(img)

        new_file = re.get_filename(output, '', file)
        cv2.imwrite(path.join(output, new_file), rot_img)
