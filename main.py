import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from test import test
from configs import get_configs
from utils import get_dataset, adjust_learning_rate, SigmoidCrossEntropyLoss, \
	generate_flip_grid

import matplotlib.pyplot as plt
import numpy as np

import sys
import argparse
import math
import time
import os


def get_parser():
	parser = argparse.ArgumentParser(description = 'CNN Attention Consistency')
	parser.add_argument("--dataset", default="wider", type=str,
		help="select a dataset to train models")
	parser.add_argument("--arch", default="resnet50", type=str,
		help="ResNet architecture")

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
	parser.add_argument('--decay', default=0.5, type=float,
		help = 'update learning rate by a factor')

	parser.add_argument('--model_dir',
		default = '/Storage/models/tmp',
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

	return parser


def main():
	parser = get_parser()
	print(parser)
	args = parser.parse_args()
	print(args)

	# load data
	opts = get_configs(args.dataset)
	print(opts)
	pos_ratio = torch.FloatTensor(opts["pos_ratio"])
	w_p = (1 - pos_ratio).exp().cuda()
	w_n = pos_ratio.exp().cuda()

	trainset, testset = get_dataset(opts)

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

	# prefix of saved checkpoint
	model_prefix = args.model_dir + '/' + args.model_prefix


	# define the model: use ResNet50 as an example
	if args.arch == "resnet50":
		from resnet import resnet50
		model = resnet50(pretrained=True, num_labels=opts["num_labels"])
		model_prefix = model_prefix + "_resnet50"
	elif args.arch == "resnet101":
		from resnet import resnet101
		model = resnet101(pretrained=True, num_labels=opts["num_labels"])
		model_prefix = model_prefix + "_resnet101"
	else:
		raise NotImplementedError("To be implemented!")

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
		raise NotImplementedError("Not supported yet!")

	# training the network
	model.train()

	# attention map size
	w1 = 7
	h1 = 7
	grid_l = generate_flip_grid(w1, h1)

	w2 = 6
	h2 = 6
	grid_s = generate_flip_grid(w2, h2)

	# least common multiple
	lcm = w1 * w2


	criterion = SigmoidCrossEntropyLoss
	criterion_mse = nn.MSELoss(size_average = True)
	for epoch in range(args.start_epoch, args.epoch_max):
		epoch_start = time.clock()
		if not args.stepsize == 0:
			adjust_learning_rate(optimizer, epoch, args)
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
			loss = criterion(output, labels, w_p, w_n)

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
			hm_l = F.upsample(hm_l, lcm)
			hm_s = F.upsample(hm_s, lcm)
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

			model_file = model_prefix + '_epoch{}.pth'
			print("Saving model to " + model_file.format(epoch+1))
			torch.save(model, model_file.format(epoch+1))

			if args.test:
				model.eval()
				test_start = time.clock()
				test(model, test_loader, epoch+1)
				test_time = (time.clock() - test_start)
				print("test time: ", test_time)
				model.train()

	final_model =model_prefix + '_final.pth'
	print("Saving model to " + final_model)
	torch.save(model, final_model)
	model.eval()
	test(model, test_loader, epoch+1)



if __name__ == '__main__':
	main()
