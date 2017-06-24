from os import path, listdir
from sys import argv

folder = argv[1]
output = argv[2]
prefix = argv[3] if len(argv) == 4 else folder

count = 0
info = open(output, 'w+')
for f in listdir(folder):
	if f[0] == '.': continue
	info.write('%s%s\n' % (prefix, f))
	count += 1

info.close()

print('file count: %d' % count)