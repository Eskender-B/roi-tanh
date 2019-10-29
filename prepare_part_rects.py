import cv2
import numpy as np
import os
from skimage import io
import matplotlib.pyplot as plt

def get_largest(im, n):
	# Find contours of the shape
	major = cv2.__version__.split('.')[0]
	if major == '3':
		_, contours, _ = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	else:
		contours, _ = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Cycle through contours and add area to array
	areas = []
	for c in contours:
		areas.append(cv2.contourArea(c))

	# Sort array of areas by size
	sorted_areas = sorted(zip(areas, contours), key=lambda x: x[0], reverse=True)

	if sorted_areas and len(sorted_areas) >= n:
		# Find nth largest using data[n-1][1]
		return sorted_areas[n - 1][1]
	else:
		return None


def load(root_dir, txt_file):

	name_list = np.loadtxt(os.path.join(root_dir, txt_file), dtype='str', delimiter=',')
	rects_list = []

	for idx in range(name_list.shape[0]):
		img_name = os.path.join(root_dir, 'images',
					name_list[idx, 1].strip() + '.jpg')

		image = np.array(io.imread(img_name), dtype=np.float)

		label_name = os.path.join(root_dir, 'labels',
			name_list[idx, 1].strip(), name_list[idx, 1].strip() + '_lbl%.2d.png')

		labels = []
		for i in [2,3,4,5,6,7,8,9]:
			labels.append(cv2.imread(label_name%i, cv2.IMREAD_GRAYSCALE))
		#labels = np.array(labels, dtype=np.float)


		# Combine mouth parts
		#labels = np.concatenate((labels[0:5], [np.clip(labels[5]+labels[6]+labels[7], 0., 255.)]), axis=0)
		mouth = np.clip(labels[5] + labels[6] + labels[7], 0, 255)
		labels = labels[0:5]
		labels.append(mouth)


		sub_list = []
		for lbl in labels:
			x,y,w,h = cv2.boundingRect(get_largest(lbl, 1))
			sub_list.extend([x,y,x+w,y+h])

			## Code check
			#lbl = cv2.rectangle(lbl, (x, y), (x+w, y+h), (255,0,0), 2)
			#plt.imshow(lbl)
			#plt.show()


		
		rects_list.append(sub_list)


	name_rects_list = np.concatenate((name_list, rects_list), axis=1)
	np.savetxt(os.path.join(root_dir, 'with_rects_'+txt_file), name_rects_list, fmt='%s', delimiter=',')


root_dir='data/SmithCVPR2013_dataset_resized'

for file in ['exemplars.txt', 'tuning.txt', 'testing.txt']:
	load(root_dir, file)