import cv2
import numpy as np
import os
from skimage import io
import matplotlib.pyplot as plt
import shutil
from mtcnn.mtcnn import MTCNN
from skimage import io, transform


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


class Warp(object):
	"""Warp class"""
	def __init__(self, landmarks):
		self.landmarks = landmarks

		src = np.array(self.landmarks)
		dst = np.array([[-0.25,-0.1], [0.25, -0.1], [0.0, 0.1], [-0.15, 0.4], [0.15, 0.4]])
		self.tform = transform.estimate_transform('similarity', src, dst)


	def warp(self, img):
		def map_func1(coords):
			tform2 = transform.SimilarityTransform(scale=1./257., rotation=0, translation=(-0.99, -0.99))
			return self.tform.inverse(np.arctanh(tform2(coords)))

		warped = transform.warp(img, inverse_map=map_func1, output_shape=[512,512] )
		return warped

	def inverse(self, warped, output_shape):
		def map_func2(coords):
			tform2 = transform.SimilarityTransform(scale=257., rotation=0, translation=(255.5, 255.5))
			return tform2(np.tanh(tform(coords)))

		warped_inv = transform.warp(warped, inverse_map=map_func2, output_shape=output_shape )
		return warped_inv


def prepare(root_dir, new_dir, txt_file):

	name_list = np.loadtxt(os.path.join(root_dir, txt_file), dtype='str', delimiter=',')
	rects_list = []
	lmarks_list = []

	for idx in range(name_list.shape[0]):
		img_name = os.path.join(root_dir, 'images',
					name_list[idx, 1].strip() + '.jpg')

		image = io.imread(img_name)

		label_name = os.path.join(root_dir, 'labels',
			name_list[idx, 1].strip(), name_list[idx, 1].strip() + '_lbl%.2d.png')


		labels = []
		for i in range(11):
			labels.append(io.imread(label_name%i))
		labels = np.array(labels)
		


		"""
		labels = []
		for i in [2,3,4,5,6,7,8,9]:
			labels.append(cv2.imread(label_name%i, cv2.IMREAD_GRAYSCALE))
		#labels = np.array(labels, dtype=np.float)


		# Combine mouth parts
		#labels = np.concatenate((labels[0:5], [np.clip(labels[5]+labels[6]+labels[7], 0., 255.)]), axis=0)
		mouth = np.clip(labels[5] + labels[6] + labels[7], 0, 255)
		labels = labels[0:5]
		labels.append(mouth)
		"""

		## Warp Image
		detector = MTCNN()
		landmarks = detector.detect_faces(image)[0]['keypoints']
		landmarks = np.array([landmarks[key] for key in ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']])
		warp_obj = Warp(landmarks)
		image =  np.uint8(warp_obj.warp(image)*255)
		labels = np.uint8(warp_obj.warp(labels.transpose(1,2,0))*255).transpose(2,0,1)

		## Save warped image and label
		img_name = os.path.join(new_dir, 'images',
					name_list[idx, 1].strip() + '.jpg')
		io.imsave(img_name, image, quality=100)

		shutil.os.mkdir(shutil.os.path.join(new_dir, 'labels', name_list[idx, 1].strip()))
		label_name = shutil.os.path.join(new_dir, 'labels', name_list[idx, 1].strip(), name_list[idx, 1].strip() + '_lbl%.2d.png')
		for i in range(len(labels)):
			io.imsave(label_name%i, labels[i], check_contrast=False)



		## Calculate part rects on warped image
		rects = []
		for i in [2,3,4,5,6]:
			x,y,w,h = cv2.boundingRect(get_largest(labels[i], 1))
			rects.extend([x,y,x+w,y+h])

		mouth = np.clip(labels[7] + labels[8] + labels[9], 0, 255)
		x,y,w,h = cv2.boundingRect(get_largest(mouth, 1))
		rects.extend([x,y,x+w,y+h])

		## Code check
		#lbl = cv2.rectangle(mouth, (x, y), (x+w, y+h), (255,0,0), 2)
		#plt.imshow(lbl)
		#plt.show()

		
		rects_list.append(rects)
		lmarks_list.append(landmarks)
		print(txt_file + ' :', idx)


	name_rects_lmarks_list = np.concatenate((name_list, rects_list, lmarks_list), axis=1)
	np.savetxt(os.path.join(new_dir, txt_file), name_rects_lmarks_list, fmt='%s', delimiter=',')


root_dir='data/SmithCVPR2013_dataset_resized'
new_dir='data/SmithCVPR2013_dataset_warped'

# Clean first
if shutil.os.path.exists(new_dir):
  shutil.rmtree(new_dir)
shutil.os.mkdir(new_dir)

shutil.os.mkdir(new_dir+'/images')
shutil.os.mkdir(new_dir+'/labels')

for file in ['exemplars.txt', 'tuning.txt', 'testing.txt']:
	prepare(root_dir, new_dir, file)