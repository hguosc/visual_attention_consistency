import os
import sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from os.path import join
from PIL import Image

from pycocotools.coco import COCO
import pylab
import random

def default_loader(path):
	return Image.open(path).convert('RGB')

def imshow(inp, title=None):
	"""Imshow for Tensor."""
	inp = inp.numpy().transpose((1, 2, 0))
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	inp = std * inp + mean
	plt.imshow(inp)
	plt.show()
	if title is not None:
		plt.title(title)


size_set = [256, 224, 192, 168, 128]

class MSCOCO(data.Dataset):

	def __init__(self, dataDir, dataType, phase):
		# self.annoDir = join(dataDir, 'annotations')
		self.imgDir = join(dataDir, dataType)
		self.phase = phase
		annoFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
		coco = COCO(annoFile)

		imgFiles = []
		imgLabels = []

		# get all image ids
		imgIds = coco.getImgIds()
		n_img = len(imgIds)
		catIds = coco.getCatIds()

		for i in range(n_img):
			img_id = imgIds[i]
			# load img info
			img_info = coco.loadImgs(img_id)[0]
			img_name = img_info['file_name']
			img_file = join(self.imgDir, img_name)
			imgFiles.append(img_file)

			# get labels
			labels = {}
			for c in catIds:
				labels[c] = 0

			# [v for k,v in labels.items()]
			annIds = coco.getAnnIds(imgIds = img_id)
			for j in range(len(annIds)):
				ann = coco.loadAnns(annIds[j])[0]
				cat_id = ann['category_id']
				labels[cat_id] = 1
				# catCount[cat_id] += 1
			imgLabels.append([v for k, v in labels.items()])

		# for k, v in catCount.items():
		#     print('{}:  {}'.format(k, v))
		self.imgFiles = imgFiles
		self.imgLabels = imgLabels

		self.transform = transforms.Compose([
			# transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean = [0.485, 0.456, 0.406],
				std = [0.229, 0.224, 0.225]),
		])

	def __getitem__(self, idx):
		imgFile = self.imgFiles[idx]
		imgLabel = self.imgLabels[idx]

		img = default_loader(imgFile)

		img1 = img.resize((288, 288))
		img2 = img.resize((256, 256))

		# imshow(image)
		# imshow(image2)
		image_lo = self.transform(img1)
		# large flip
		image_lf = self.transform(img1.transpose(Image.FLIP_LEFT_RIGHT))

		# small orig
		image_so = self.transform(img2)
		# small flip
		image_sf = self.transform(img2.transpose(Image.FLIP_LEFT_RIGHT))

		labels = torch.FloatTensor(imgLabel)
		# print sample['file_name']
		# print labels
		# imshow(image)
		# sys.exit()
		return image_lo, image_lf, image_so, image_sf, labels


	def __len__(self):
		return len(self.imgFiles)


def getSubsets(dataDir):
	trainset = MSCOCO(dataDir, 'train2014', 'train')
	valset = MSCOCO(dataDir, 'val2014', 'test')
	return trainset, valset


if __name__ == '__main__':
	dataDir = '/data/hguo/Datasets/MS-COCO'
	dataType = 'val2014'
	mscoco = MSCOCO(dataDir, dataType, 'test')
	mscoco[0]
	mscoco[1]
