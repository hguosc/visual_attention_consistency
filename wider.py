import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from os.path import exists, join, basename
from PIL import Image

crop_size = 224
scale_size = 224

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

class WiderAttr(data.Dataset):
	def __init__(self, subset, anno_dir, data_dir):
		self.subset = subset
		self.data_dir = data_dir
		if self.subset == 'train':
			anno_file = join(anno_dir, 'wider_attribute_trainval.json')
		else:
			anno_file = join(anno_dir, 'wider_attribute_test.json')

		self.transform = transforms.Compose([
			# transforms.CenterCrop((crop_size, crop_size)),
			transforms.ToTensor(),
			transforms.Normalize(mean = [0.485, 0.456, 0.406],
				std = [0.229, 0.224, 0.225]),
			])

		with open(anno_file) as AF:
			anno = json.load(AF)

		self.images = anno['images']
		self.attributes = anno['attribute_id_map']
		self.scenes = anno['scene_id_map']

		# create a dict to store separate bboxes
		samples = {}
		num_img = len(self.images)
		s_id = 0
		for i in range(num_img):
			file_name = self.images[i]['file_name']#.encode('utf-8')
			scene_id = self.images[i]['scene_id']
			targets = self.images[i]['targets']
			num_tar = len(targets)

			for j in range(num_tar):
				attribute = targets[j]['attribute']
				bbox = targets[j]['bbox']
				samples[s_id] = {}
				samples[s_id]['file_name'] = file_name
				samples[s_id]['scene_id'] = scene_id
				samples[s_id]['labels'] = attribute
				samples[s_id]['bbox'] = bbox
				s_id += 1

				# img_file = join(data_dir, file_name)
				# img = default_loader(img_file)
				# wd, ht = img.size
				# if bbox[0] > wd or bbox[1] > ht:
				# 	print file_name
				# 	print bbox

		self.samples = samples


	def __getitem__(self, idx):
		# sampe: self.samples[idx]
		sample = self.samples[idx]
		img_file = join(self.data_dir, sample['file_name'])
		labels = sample['labels']
		bbox = sample['bbox']
		scene_id = sample['scene_id']

		# load image
		img = default_loader(img_file)
		wd, ht = img.size

		# crop bounding box
		# bbox: x, y, w, h -- need to be x1, y1, x2, y2
		# extend
		x = bbox[0]
		y = bbox[1]
		w = bbox[2]
		h = bbox[3]

		bbox[2] = x+w
		bbox[3] = y+h

		# there are some samples not annotated well
		if x > wd or y > ht:
			bbox = [0, 0, wd, ht]

		img_crop = img.crop(tuple(bbox))
		t1,t2 = img_crop.size
		if t1 == 0. or t2 == 0.:
			# find if there still images not work
			print(sample)

		img1 = img_crop.resize((224, 224))
		img2 = img_crop.resize((192, 192))

		# large orig
		image_lo = self.transform(img1)
		# large flip
		image_lf = self.transform(img1.transpose(Image.FLIP_LEFT_RIGHT))

		# small orig
		image_so = self.transform(img2)
		# small flip
		image_sf = self.transform(img2.transpose(Image.FLIP_LEFT_RIGHT))

		labels = torch.FloatTensor(labels)
		# print sample['file_name']
		# print labels
		# imshow(image)
		# import pdb; pdb.set_trace()
		return image_lo, image_lf, image_so, image_sf, labels

	def __len__(self):
		return len(self.samples)


def get_subsets(anno_dir, data_dir):
	trainset = WiderAttr('train', anno_dir, data_dir)
	testset = WiderAttr('test', anno_dir, data_dir)
	return trainset, testset

if __name__ == '__main__':
	subset = 'test'
	anno_dir = '/path/to/wider_attribute_annotation'
	data_dir = '/path/to/Image'
	wa = WiderAttr(subset, anno_dir, data_dir)
	trainset, testset = get_subsets(anno_dir, data_dir)
	wa[1]

	train_loader = torch.utils.data.DataLoader(
		trainset,
		batch_size = 32,
		shuffle = True,
		num_workers = 8,
		)
	test_loader = torch.utils.data.DataLoader(
		testset,
		batch_size = 1,
		shuffle = False,
		num_workers = 2,
		)
