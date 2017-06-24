from os import path, listdir, rename
from sys import argv

import cv2

folder = argv[1]
output = argv[2]
prefix = argv[3] if len(argv) == 4 else folder

# for i, file in enumerate(listdir(folder)):
# 	old_filepath = path.join(folder, file)
# 	new_filePath = path.join(folder, ('%05d.png' % i))
# 	rename(old_filepath, new_filePath)

count = 0
info = open(output, 'w+')
for i, file in enumerate(listdir(folder)):
	filepath = path.join(folder, file)
	image = cv2.imread(filepath)
	if image is None: continue
	info.write('%s%s 1 0 0 %d %d\n' % (prefix, file, image.shape[1], image.shape[0]))
	count += 1
info.close()

print('file count: %d' % count)