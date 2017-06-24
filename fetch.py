from sys import argv
import time
import csv
from os import path, mkdir
from urllib import request
import cv2
import recognize as rcg
from utils.rotate import *
from shutil import copyfile, rmtree

letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

org_folder = argv[1]
dst_folder = argv[2]

for l in rcg.meter_labels:
    dst = path.join(dst_folder, l)
    if path.exists(dst):
        rmtree(dst)
    mkdir(dst)


with open('./data/data.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    count = 0
    for row in reader:

        count += 1

        # 跳过6位数的水表
        if len(row[1]) > 5:
            continue

        for l in letters:
            name = '%05d%s.png' % (int(row[1]), l)
            if not path.exists(path.join(org_folder, name)):
                with open(path.join(org_folder, name), 'wb') as f:
                    url = row[0].replace('file.wap.wzswjt.com', '172.16.232.23')
                    f.write(request.urlopen(url).read())
                    print(url)
                    break

        img = cv2.imread(path.join(org_folder, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        meter_mod = rcg.get_module('./output/classify/meter', 1)
        rotate_mod = rcg.get_module('./output/rotate/rotate', 5)

        # 区分是否为水表
        meter_prob, meter_label = rcg.classify(meter_mod, img, rcg.meter_labels)

        if meter_label != 'nometer':

            # 区分角度
            rotate_prob, rotate = rcg.classify(rotate_mod, img, rcg.rotate_labels)

            if rotate == '180':
                img = rotate180(img)
            elif rotate == 'L90':
                img = rotate90(img)
            elif rotate_mod == 'R90':
                img = rotate270(img)

        # copyfile(path.join(org_folder, name), path.join(dst_folder, meter_label, name))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path.join(dst_folder, meter_label, name), img)

        print('file %s belongs label %s, prob=%f, rotate=%s'% (name, meter_label, meter_prob, rotate))


        if count % 100 == 0:
            print('Total Count = %d' % count)

        # 暂停1秒
        # time.sleep(1)
