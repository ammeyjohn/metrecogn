from sys import argv
from os import path, listdir
from shutil import copyfile
try:
    from rename import *
except:
    from utils.rename import *

def merge(src, dst):
    for file in listdir(src):
        if file == '.DS_Store':
            continue

        new_name = get_filename(dst, '', file)
        copyfile(path.join(src, file), path.join(dst, new_name))


if __name__ == '__main__':

    src_folder = argv[1]
    dst_folder = argv[2]

    merge(src_folder, dst_folder)
