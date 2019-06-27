import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from torch.autograd import Variable
from wider import get_subsets
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
	default = '/path/to/WIDER/dataset/<Image>',
	type = str,
	help = 'path to "Image" folder of WIDER dataset')
parser.add_argument('--anno_dir',
	default = '/path/to/wider_attribute_annotation',
	type = str,
	help = 'annotation file')

parser.add_argument('--train_batch_size', default = 16, type = int,
	help = 'default training batch size')
parser.add_argument('--train_workers', default = 4, type = int,
	help = '# of workers used to load training samples')
parser.add_argument('--test_batch_size', default = 8, type = int,
	help = 'default test batch size')
parser.add_argument('--test_workers', default = 4, type = int,
	help = '# of workers used to load testing samples')

parser.add_argument('--learning_rate', default = 0.001, type = float,
	help = 'base learning rate')
parser.add_argument('--momentum', default = 0.9, type = float,
	help = "set the momentum")
parser.add_argument('--weight_decay', default = 0.0005, type = float,
	help = 'set the weight_decay')
parser.add_argument('--stepsize', default = 3, type = int,
	help = 'lr decay each # of epoches')

parser.add_argument('--model_dir',
	default = '/path/to/model/saving/dir',
	type = str,
	help = 'path to save checkpoints')
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
parser.add_argument('--epoch_max', default = 12, type = int,
	help = 'max # of epcoh')
parser.add_argument('--display', default = 200, type = int,
	help = 'display')
parser.add_argument('--snapshot', default = 1, type = int,
	help = 'snapshot')
parser.add_argument('--start_epoch', default = 0, type = int,
	help = 'resume training from specified epoch')
parser.add_argument('--resume', default = '', type = str,
	help = 'resume training from specified model state')

parser.add_argument('--test', default = True, type = bool,
	help = 'conduct testing after each checkpoint being saved')

# pre-calculated weights to balance positive and negative samples of each label
pos_ratio = [0.5669, 0.2244, 0.0502, 0.2260, 0.2191, 0.4647, 0.0699, 0.1542, \
	0.0816, 0.3621, 0.1005, 0.0330, 0.2682, 0.0543]
pos_ratio = torch.FloatTensor(pos_ratio)
w_p = (1 - pos_ratio).exp().cuda()
w_n = pos_ratio.exp().cuda()


def adjust_learning_rate(optimizer, epoch, stepsize):
	"""Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
	lr = args.learning_rate * (0.5 ** (epoch // stepsize))
	print("Current learning rate is: {:.5f}".format(lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def SigmoidCrossEntropyLoss(x, y):
	# weighted sigmoid cross entropy loss defined in Li et al. ACPR'15
	loss = 0.0
	if not x.size() == y.size():
		print("x and y must have the same size")
	else:
		N = y.size(0)
		L = y.size(1)
		for i in range(N):
			w = torch.zeros(L).cuda()
			w[y[i].data == 1] = w_p[y[i].data == 1]
			w[y[i].data == 0] = w_n[y[i].data == 0]

			w = Variable(w, requires_grad = False)
			temp = - w * ( y[i] * (1 / (1 + (-x[i]).exp())).log() + \
				(1 - y[i]) * ( (-x[i]).exp() / (1 + (-x[i]).exp()) ).log() )
			loss += temp.sum()

		loss = loss / N
	return loss


def generate_flip_grid(w, h):
	# used to flip attention maps
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

	# load data
	data_dir = args.data_dir
	anno_dir = args.anno_dir

	trainset, testset = get_subsets(anno_dir, data_dir)
	train_loader = torch.utils.data.DataLoader(trainset,
		batch_size = args.train_batch_size,
		shuffle = True,
		num_workers = args.train_workers)
	test_loader = torch.utils.data.DataLoader(testset,
		batch_size = args.test_batch_size,
		shuffle = False,
		num_workers = args.test_workers)


	# path to save models
	if not os.path.isdir(args.model_dir):
		print("Make directory: " + args.model_dir)
		os.makedirs(args.model_dir)

	model_prefix = args.model_dir + '/' + args.model_prefix


	# define the model: use ResNet50 as an example
	from resnet import resnet50
	model = resnet50(pretrained = True)

	if args.start_epoch != 0:
		resume_model = torch.load(args.resume)
		resume_dict = resume_model.state_dict()
		model_dict = model.state_dict()
		resume_dict = {k:v for k,v in resume_dict.items() if k in model_dict}
		model_dict.update(resume_dict)
		model.load_state_dict(model_dict)

	# print(model)
	model.cuda()

	if args.optimizer == 'Adam':
		optimizer = optim.Adam(
			model.parameters(),
			lr = args.learning_rate
		)
	elif args.optimizer == 'SGD':
		optimizer = optim.SGD(
			model.parameters(),
			lr = args.learning_rate,
			momentum = args.momentum,
			weight_decay = args.weight_decay
		)
	else:
		pass

	# training the network
	model.train()

	# attention map size
	w = 7
	h = 7
	grid_l = generate_flip_grid(w, h)

	w = 6
	h = 6
	grid_s = generate_flip_grid(w, h)


	criterion = SigmoidCrossEntropyLoss
	criterion_mse = nn.MSELoss(size_average = True)
	for epoch in range(args.start_epoch, args.epoch_max):
		epoch_start = time.clock()
		if not args.stepsize == 0:
			aedjust_learning_rate(optimizer, epoch, args.stepsize)
		for step, batch_data in enumerate(train_loader):
			batch_images_lo = batch_data[0]
			batch_images_lf = batch_data[1]
			batch_images_so = batch_data[2]
			batch_images_sf = batch_data[3]
			batch_labels = batch_data[4]

			batch_labels[batch_labels == -1] = 0

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
			hm_l = F.upsample(hm_l, 42)
			hm_s = F.upsample(hm_s, 42)
			scale_loss = F.mse_loss(hm_l, hm_s)

			losses = loss + flip_loss_l + flip_loss_s + scale_loss

			optimizer.zero_grad()
			losses.backward()
			optimizer.step()

			if (step) % args.display == 0:
				print(
					'epoch: {},\ttrain step: {}\tLoss: {:.6f}'.format(epoch+1,
					step, losses.data[0])
				)
				print(
					'\tcls loss: {:.4f};\tflip_loss_l: {:.4f}'
					'\tflip_loss_s: {:.4f};\tscale_loss: {:.4f}'.format(
						loss.data[0], 
						flip_loss_l.data[0], 
						flip_loss_s.data[0], 
						scale_loss.data[0]
					)
				)

		epoch_end = time.clock()
		elapsed = epoch_end - epoch_start
		print("Epoch time: ", elapsed)

		# test
		if (epoch+1) % args.snapshot == 0:

			model_file = model_prefix + '_resnet50_{}.pth'
			print("Saving model to " + model_file.format(epoch+1))
			torch.save(model, model_file.format(epoch+1))

			if args.test:
				model.eval()
				test_start = time.clock()
				test(model, test_loader, epoch+1)
				test_time = (time.clock() - test_start)
				print("test time: ", test_time)
				model.train()

	final_model =model_prefix + '_resnet50_final.pth'
	print("Saving model to " + final_model)
	torch.save(model, final_model)
	model.eval()
	test(model, test_loader, epoch+1)



if __name__ == '__main__':
	main()
