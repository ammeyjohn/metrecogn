from sys import argv
import random
from os import path, listdir, rename, remove

if __name__ == '__main__':

    src = argv[1]
    output = argv[2]
    ratio = float(argv[3]) if len(argv) >= 4 else 0.95

    train_file = open(output + '_train.lst', '+w')
    test_file = open(output + '_test.lst', '+w')

    for i, file in enumerate(listdir(src)):
        if file == '.DS_Store':
            continue

        name, ext = path.splitext(file)
        label = int(name[:-1])
        line = '%d\t%d\t%s\n' % (i, label, file)

        rnd_val = random.random()
        if rnd_val >= ratio:
            test_file.write(line)
        else:
            train_file.write(line)

    train_file.close()
    test_file.close()