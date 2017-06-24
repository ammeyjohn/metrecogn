# -*- coding: utf-8 -*-

from sys import argv
from os import listdir, remove, path
from PIL import Image
import imagehash

src_folder = argv[1]

dic = {}


count = 0
for src_file in listdir(src_folder):
    if src_file == '.DS_Store':
        continue

    image = Image.open(path.join(src_folder, src_file))
    h = str(imagehash.dhash(image))

    if h not in dic:
        dic[h] = []
    dic[h].append(path.join(src_folder, src_file))

    count += 1
    if count % 100 == 0:
        print(count)

for key, files in dic.items():
    for file in files[1:]:
        if path.exists(file):
            remove(file)
            print('file %s removed, md5=%s' % (file, key))
