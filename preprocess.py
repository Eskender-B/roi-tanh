from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from datetime import datetime
from mtcnn.mtcnn import MTCNN
import cv2
import pickle

def get_largest(im, n):
	# Find contours of the shape

	im = np.uint8(im)

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



class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		image, labels, rects, idx = sample['image'], sample['labels'], sample['rects'], sample['index']

		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		image = image.transpose((2, 0, 1))
		return {'image': torch.from_numpy(image).float(),
		'labels': torch.from_numpy(labels).float(),
		'rects': torch.from_numpy(rects).float(),
		'index': idx}



class Warp(object):
	"""Warp class"""
	def __init__(self, landmarks):
		self.landmarks = landmarks

		src = np.array([self.landmarks['left_eye'], self.landmarks['right_eye'], self.landmarks['nose'], self.landmarks['mouth_left'], self.landmarks['mouth_right']])
		dst = np.array([[-0.25,-0.1], [0.25, -0.1], [0.0, 0.1], [-0.15, 0.4], [0.15, 0.4]])
		self.tform = transform.estimate_transform('similarity', src, dst)


	def warp(self, img):
		def map_func1(coords):
			tform2 = transform.SimilarityTransform(scale=1./256., rotation=0, translation=(-1.0, -1.0))
			return self.tform.inverse(np.arctanh(tform2(coords)))

		warped = transform.warp(img, inverse_map=map_func1, output_shape=[512,512] )
		return warped

	def inverse(self, warped, output_shape):
		def map_func2(coords):
			tform2 = transform.SimilarityTransform(scale=256., rotation=0, translation=(255.5, 255.5))
			return tform2(np.tanh(tform(coords)))

		warped_inv = transform.warp(warped, inverse_map=map_func2, output_shape=output_shape )
		return warped_inv




class ImageDataset(Dataset):
	"""Image dataset."""
	def __init__(self, txt_file, root_dir, bg_indexs=set([]), fg_indexs=None, transform=None):
		"""
		Args:
		txt_file (string): Path to the txt file with list of image id, name.
		root_dir (string): Directory with all the images.
		transform (callable, optional): Optional transform to be applied
		on a sample.
		"""
		self.name_list = np.loadtxt(os.path.join(root_dir, txt_file), dtype='str', delimiter=',')
		self.root_dir = root_dir
		self.transform = transform

		if not fg_indexs:
			self.bg_indexs = sorted(bg_indexs)
			self.fg_indexs = sorted(set(range(11)).difference(bg_indexs))
		else:
			self.fg_indexs = sorted(fg_indexs)

	def __len__(self):
		return len(self.name_list)

	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir, 'images',
			self.name_list[idx, 1].strip() + '.jpg')

		image = np.array(io.imread(img_name), dtype=np.float)

		label_name = os.path.join(self.root_dir, 'labels',
			self.name_list[idx, 1].strip(), self.name_list[idx, 1].strip() + '_lbl%.2d.png')

		labels = []
		for i in self.fg_indexs:
			labels.append(io.imread(label_name%i))
		
		labels = np.array(labels, dtype=np.float)
		#labels = np.concatenate((labels, [255.0-labels.sum(0)]), axis=0)
				

		# Add background
		if type(labels).__module__==np.__name__:
			labels = np.concatenate((labels, [255-labels.sum(0)]), axis=0)
		else:
			labels = torch.cat([labels, torch.tensor(255).to(labels.device) - labels.sum(0, keepdim=True)], 0)


		## Warp object
		detector = MTCNN()
		landmarks = detector.detect_faces(image)[0]['keypoints']
		warp_obj = Warp(landmarks)
		image, labels=  warp_obj.warp(image), warp_obj.warp(labels)



		## Calculate part rects on warped image
		rects = []
		for i in [2,3,4,5,6]:
			x,y,w,h = cv2.boundingRect(get_largest(labels[i], 1))
			rects.extend([x,y,x+w,y+h])

		mouth = np.clip(labels[7] + labels[8] + labels[9], 0, 255)
		x,y,w,h = cv2.boundingRect(get_largest(mouth, 1))
		rects.extend([x,y,x+w,y+h])

		rects = np.array(rects)


		sample = {'image': image, 'labels': labels, 'rects':rects, 'index':idx}

		if self.transform:
			sample = self.transform(sample)

		sample['landmarks'] = landmarks

		return sample