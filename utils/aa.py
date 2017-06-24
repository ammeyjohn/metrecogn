

from os import path, rename


basedir = 'images/original/labelled'
with open('data/rename.txt', 'r') as f:
	lines = f.readlines()
	for l in lines:
		dst, src = l.strip().split(' ')
		rename(path.join(basedir, src), path.join(basedir, dst))