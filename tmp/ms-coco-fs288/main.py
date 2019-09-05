# this script should be the entrance for following coding
# together with wider.py test.py, and specific resnet.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from torch.autograd import Variable
from ms_coco import getSubsets
from test import test

import matplotlib.pyplot as plt
import numpy as np

import sys
import argparse
import math
import time
import os


parser = argparse.ArgumentParser(description = 'Attribute')

parser.add_argument('--data_dir',
	default = '/work/hguo/datasets/MS-COCO',
	type = str,
	help = 'path to "Image" folder of WIDER dataset')
# parser.add_argument('--anno_file',
# 	default = '/data/hguo/Datasets/PA-100K/annotation/annotation.mat',
# 	type = str,
# 	help = 'annotation file')

parser.add_argument('--train_batch_size', default = 24, type = int,
	help = 'default training batch size')
parser.add_argument('--train_workers', default = 12, type = int,
	help = '# of workers used to load training samples')
parser.add_argument('--test_batch_size', default = 8, type = int,
	help = 'default test batch size')
parser.add_argument('--test_workers', default = 12, type = int,
	help = '# of workers used to load testing samples')

parser.add_argument('--learning_rate', default = 0.001, type = float,
	help = 'base learning rate')
parser.add_argument('--momentum', default = 0.9, type = float,
	help = "set the momentum")
parser.add_argument('--weight_decay', default = 0.0005, type = float,
	help = 'set the weight_decay')
parser.add_argument('--stepsize', default = 6, type = int,
	help = 'lr decay each # of epoches')

parser.add_argument('--model_dir',
	default = '/work/hguo/models/ac_mscoco/mscoco_fs288',
	type = str,
	help = 'path to save trained models')
parser.add_argument('--model_prefix',
	default = 'model',
	type = str,
	help = 'model file name starts with')

# optimizer
parser.add_argument('--optimizer',
	default = 'SGD',
	type = str,
	help = 'Select an optimizer: TBD')


# general parameters
parser.add_argument('--epoch_max', default = 14, type = int,
	help = 'max # of epcoh')
parser.add_argument('--display', default = 200, type = int,
	help = 'display')
parser.add_argument('--snapshot', default = 1, type = int,
	help = 'snapshot')
parser.add_argument('--resume', default = 0, type = int,
	help = 'resume training from specified epoch')


import scipy.io
ratio_file = '../ratio.mat'
temp = scipy.io.loadmat(ratio_file)
ratio = temp['ratio']
ratio = torch.from_numpy(ratio).squeeze().type(torch.FloatTensor)
w_p = (1-ratio).exp().cuda()
w_n = ratio.exp().cuda()


def adjust_learning_rate(optimizer, epoch, stepsize):
	"""Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
	lr = args.learning_rate * (0.1 ** (epoch // stepsize))
	# print('6, 12, ...')
	# if epoch < 6:
	# 	lr = 0.001
	# else:
	# 	if epoch < 12:
	# 		lr = 0.0001
	# 	else:
	# 		lr = 0.00001
	print("Current learning rate is: {:.6f}".format(lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

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

def multi_label_classification_loss(x, y):

	loss = 0.0
	if not x.size() == y.size():
		print("x and y must have the same size")
	else:

		for i in range(y.size(0)):
			temp = -(y[i]*y[i] * ( 1 / (1+(-x[i]*y[i]).exp()) ).log())
			loss += temp.sum()

		loss = loss / y.size(0)
	return loss


def SigmoidCrossEntropyLoss(x, y):
	loss = 0.0
	if not x.size() == y.size():
		print("x and y must have the same size")
	else:
		N = y.size(0)
		L = y.size(1)
		for i in range(N):
			w = torch.zeros(L).cuda()
			# print(y[i].data)
			# print(w_p)
			# print(w_n)
			w[y[i].data == 1] = w_p[y[i].data == 1]
			w[y[i].data == 0] = w_n[y[i].data == 0]
			# sys.exit()

			w = Variable(w, requires_grad = False)
			temp = - w * ( y[i] * (1 / (1 + (-x[i]).exp())).log() + \
				(1 - y[i]) * ( (-x[i]).exp() / (1 + (-x[i]).exp()) ).log() )
			loss += temp.sum()

		loss = loss / N
	return loss

def generate_flip_grid(w, h):

	x_ = torch.arange(w).view(1, -1).expand(h, -1)
	y_ = torch.arange(h).view(-1, 1).expand(-1, w)
	grid = torch.stack([x_, y_], dim=0).float().cuda()
	grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
	grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
	grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1

	grid[:, 0, :, :] = -grid[:, 0, :, :]
	return grid


def main():
	global args
	args = parser.parse_args()
	print(args)

	num_cls = 80

	# load data
	data_dir = args.data_dir

	trainset, valset = getSubsets(data_dir)
	train_loader = torch.utils.data.DataLoader(trainset,
		batch_size = args.train_batch_size,
		shuffle = True,
		num_workers = args.train_workers)
	test_loader = torch.utils.data.DataLoader(valset,
		batch_size = args.test_batch_size,
		shuffle = False,
		num_workers = args.test_workers)


	# path to save models
	if not os.path.isdir(args.model_dir):
		print("Make directory: " + args.model_dir)
		os.makedirs(args.model_dir)

	model_prefix = args.model_dir + '/' + args.model_prefix


	# define the model
	from resnet import resnet101
	model = resnet101(pretrained = True)
#	resume_model_file = '/data/userdata/hguo/models/AttrFlow/mscoco_flow_scale_wsce_288_b16/model_resnet101_4.pth'
#	resume_model = torch.load(resume_model_file)
#	resume_dict = resume_model.state_dict()
#	model_dict = model.state_dict()
#	resume_dict = {k:v for k,v in resume_dict.items() if k in model_dict}
#	model_dict.update(resume_dict)
#	model.load_state_dict(model_dict)
	resume_epoch = 0
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)

	model.cuda()


	if args.optimizer == 'Adam':
		optimizer = optim.Adam(
			model.parameters(),
			lr = args.learning_rate)
	elif args.optimizer == 'SGD':
		optimizer = optim.SGD(
			model.parameters(),
			lr = args.learning_rate,
			momentum = args.momentum,
			weight_decay = args.weight_decay)
	else:
		pass

	# training the network
	model.train()

	w = 9
	h = 9
	grid_l = generate_flip_grid(w, h)

	w = 8
	h = 8
	grid_s = generate_flip_grid(w, h)

	# criterion = multi_label_classification_loss
	# criterion = nn.MultiLabelSoftMarginLoss()
	criterion = SigmoidCrossEntropyLoss

	for epoch in range(resume_epoch, args.epoch_max):
		epoch_start = time.clock()
		if not args.stepsize == 0:
			adjust_learning_rate(optimizer, epoch, args.stepsize)
		for step, batch_data in enumerate(train_loader):
			batch_images_lo = batch_data[0]
			batch_images_lf = batch_data[1]
			batch_images_so = batch_data[2]
			batch_images_sf = batch_data[3]
			batch_labels = batch_data[4]

			if batch_labels.size(0) != args.train_batch_size:
				continue

			batch_images_l = torch.cat((batch_images_lo, batch_images_lf))
			batch_images_s = torch.cat((batch_images_so, batch_images_sf))
			batch_labels = torch.cat((batch_labels, batch_labels, batch_labels, batch_labels))

			batch_images_l = batch_images_l.cuda()
			batch_images_s = batch_images_s.cuda()
			batch_labels = batch_labels.cuda()

			inputs_l = Variable(batch_images_l)
			inputs_s = Variable(batch_images_s)
			labels = Variable(batch_labels)

			output_l, hm_l = model(inputs_l)
			output_s, hm_s = model(inputs_s)

			output = torch.cat((output_l, output_s))
			loss = criterion(output, labels)


			# flip
			num = hm_l.size(0) // 2

			hm1, hm2 = hm_l.split(num)
			flip_grid_large = grid_l.expand(num, -1, -1, -1)
			flip_grid_large = Variable(flip_grid_large, requires_grad = False)
			flip_grid_large = flip_grid_large.permute(0, 2, 3, 1)
			hm2_flip = F.grid_sample(hm2, flip_grid_large, mode = 'bilinear',
				padding_mode = 'border')
			flip_loss_l = F.mse_loss(hm1, hm2_flip)

			hm1_small, hm2_small = hm_s.split(num)
			flip_grid_small = grid_s.expand(num, -1, -1, -1)
			flip_grid_small = Variable(flip_grid_small, requires_grad = False)
			flip_grid_small = flip_grid_small.permute(0, 2, 3, 1)
			hm2_small_flip = F.grid_sample(hm2_small, flip_grid_small, mode = 'bilinear',
				padding_mode = 'border')
			flip_loss_s = F.mse_loss(hm1_small, hm2_small_flip)

			# scale loss
			num = hm_l.size(0)
			hm_l = F.upsample(hm_l, 72)
			hm_s = F.upsample(hm_s, 72)
			scale_loss = F.mse_loss(hm_l, hm_s)

			losses = loss + 0.6 * flip_loss_l + 0.6 * flip_loss_s + 0.8 * scale_loss

			optimizer.zero_grad()
			losses.backward()
			optimizer.step()

			if (step) % args.display == 0:
				print('epoch: {},\ttrain step: {}\tLoss: {:.6f}'.format(epoch+1,
					step, losses.data[0]))
				print('cls loss: {:.5f}'.format(loss.data[0]))
				print('flip_loss_l: {:.5f}'.format(flip_loss_l.data[0]))
				print('flip_loss_s: {:.5f}'.format(flip_loss_s.data[0]))
				print('scale_loss: {:.5f}'.format(scale_loss.data[0]))

		epoch_end = time.clock()
		elapsed = epoch_end - epoch_start
		print("Epoch time: ", elapsed)


		# test
		if (epoch+1) % args.snapshot == 0:

			model_file = model_prefix + '_resnet101_{}.pth'
			print("Saving model to " + model_file.format(epoch+1))
			torch.save(model, model_file.format(epoch+1))

			model.eval()
			test_start = time.clock()
			test(model, test_loader, epoch+1)
			test_time = (time.clock() - test_start)
			print("test time: ", test_time)
			model.train()

	final_model =model_prefix + '_resnet101_final.pth'
	print("Saving model to " + final_model)
	torch.save(model, final_model)
	model.eval()
	# test(model, test_loader)



if __name__ == '__main__':
	main()
