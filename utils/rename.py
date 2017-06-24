from sys import argv
from os import path, listdir, rename, remove
import utils

letters = list('ABCDEFGHIJKLMNOPQQRSTUVWXYZ')


def get_filename(folder, prefix, ori_name):
    name, ext = path.splitext(ori_name)
    if len(name) > 6 and utils.is_alphabet(name[0]):
        return ori_name
    for l in letters:
        new_name = prefix + name[:-1] + l + ext
        if not path.exists(path.join(folder, new_name)):
            return new_name


seq = 0


def get_seq_filename(folder, prefix, ori_name):
    global seq
    name, ext = path.splitext(ori_name)
    while True:
        seq += 1
        new_name = prefix + ('%010d' % seq) + ext
        if not path.exists(path.join(folder, new_name)):
            return new_name


def file_rename():
    folder = argv[1]
    prefix = argv[2] if len(argv) >= 3 else ''

    count = 0
    for file in listdir(folder):
        if file == '.DS_Store':
            continue

        if len(file) > 20:
            # remove(path.join(folder, file))
            continue

        if utils.is_number(file[0]):
            continue

        new_name = get_filename(folder, prefix, file)
        print('rename %s -> %s' % (file, new_name))
        rename(path.join(folder, file), path.join(folder, new_name))

        count += 1

    print('Rename file count: %d' % count)


if __name__ == '__main__':

    file_rename()
