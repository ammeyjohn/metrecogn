import os


src_folder = './images/classified/meters'
dst_folder = './images/classified/WS'


for dst_file in os.listdir(dst_folder):
	if os.path.exists(os.path.join(src_folder, dst_file)):
		os.remove(os.path.join(src_folder, dst_file))
		print('file %s removed' % dst_file)	