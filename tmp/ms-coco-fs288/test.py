import torch
import torch.nn as nn

import torchvision.models as models

from torch.autograd import Variable

import numpy as np
import sys
import math
import time
import matplotlib.pyplot as plt

import sklearn.metrics as metrics
import scipy.io

from sklearn.metrics import average_precision_score
from ms_coco import getSubsets

num_attr = 80

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

def calc_average_precision(y_true, y_score):
	aps = np.zeros(num_attr)
	for i in range(num_attr):
		true = y_true[i]
		score = y_score[i]

		ap = average_precision_score(true, score)
		aps[i] = ap

	return aps

def calc_acc_pr_f1(y_true, y_pred):
	precision = np.zeros(num_attr)
	recall = np.zeros(num_attr)
	accuracy = np.zeros(num_attr)
	f1 = np.zeros(num_attr)
	for i in range(num_attr):
		true = y_true[i]
		pred = y_pred[i]

		precision[i] = metrics.precision_score(true, pred)
		recall[i] = metrics.recall_score(true, pred)
		accuracy[i] = metrics.accuracy_score(true, pred)
		f1[i] = metrics.f1_score(true, pred)

	return precision, recall, accuracy, f1

def calc_mean_acc(y_true, y_pred):
	macc = np.zeros(num_attr)
	for i in range(num_attr):
		true = y_true[i]	# 0, 1
		pred = y_pred[i]	# 0, 1

		temp = true + pred
		tp = (temp[temp == 2]).size
		tn = (temp[temp == 0]).size
		p = (true[true == 1]).size
		n = (true[true == 0]).size

		macc[i] = .5 * tp / (p) + .5 * tn / (n)

	return macc

def calc_acc_pr_f1_overall(y_true, y_pred):

	true = y_true
	pred = y_pred

	precision = metrics.precision_score(true, pred)
	recall = metrics.recall_score(true, pred)
	accuracy = metrics.accuracy_score(true, pred)
	f1 = metrics.f1_score(true, pred)

	return precision, recall, accuracy, f1

def calc_mean_acc_overall(y_true, y_pred):

	true = y_true	# 0, 1
	pred = y_pred	# 0, 1

	temp = true + pred
	tp = (temp[temp == 2]).size
	tn = (temp[temp == 0]).size
	p = (true[true == 1]).size
	n = (true[true == 0]).size
	macc = .5 * tp / (p) + .5 * tn / (n)

	return macc

def eval_example(y_true, y_pred):
	N = y_true.shape[1]

	acc = 0.
	prec = 0.
	rec = 0.
	f1 = 0.

	for i in range(N):
		true_exam = y_true[:,i]		# column: labels for an example
		pred_exam = y_pred[:,i]

		temp = true_exam + pred_exam

		yi = true_exam.sum()	# number of attributes for i
		fi = pred_exam.sum()	# number of predicted attributes for i
		ui = (temp > 0).sum()	# temp == 1 or 2 means the union of attributes in yi and fi
		ii = (temp == 2).sum()	# temp == 2 means the intersection

		acc += 1.0 * ii / ui
		prec += 1.0 * ii / fi
		rec += 1.0 * ii / yi

	acc /= N
	prec /= N
	rec /= N
	f1 = 2.0 * prec * rec / (prec + rec)
	return acc, prec, rec, f1



def test(model, test_loader, epoch):
	print("testing ... ")

	probs = torch.FloatTensor()
	gtruth = torch.FloatTensor()
	probs = probs.cuda()
	gtruth = gtruth.cuda()
	for i, sample in enumerate(test_loader):
		images = sample[0]
		labels = sample[4]
		labels = labels.type(torch.FloatTensor)

		images = images.cuda()
		labels = labels.cuda()

		test_input = Variable(images)
		y, _ = model(test_input)

		probs = torch.cat((probs, y.data.transpose(1, 0)), 1)
		gtruth = torch.cat((gtruth, labels.transpose(1, 0)), 1)

	print('predicting finished ....')

	preds = np.zeros((probs.size(0), probs.size(1)))
	temp = probs.cpu().numpy()
	preds[temp > 0.] = 1

	scipy.io.savemat('./preds/prediction_e{}.mat'.format(epoch), dict(gt = gtruth.cpu().numpy(), \
		prob = probs.cpu().numpy(), pred = preds))

	aps = calc_average_precision(gtruth.cpu().numpy(), probs.cpu().numpy())
	print('>>>>>>>>>>>>>>>>>>>>>>>> Average for Each Attribute >>>>>>>>>>>>>>>>>>>>>>>>>>>')
	print("APs")
	print(aps)
	precision, recall, accuracy, f1 = calc_acc_pr_f1(gtruth.cpu().numpy(), preds)
	print('precision scores')
	print(precision)
	print('recall scores')
	print(recall)
	print('f1 scores')
	print(f1)
	print('')



	macc = calc_mean_acc(gtruth.cpu().numpy(), preds)
	print('mA scores')
	print(macc)

	print("\nmean AP: {}".format(aps.mean()))
	print('F1-C: {}'.format(f1.mean()))
	print('P-C: {}'.format(precision.mean()))
	print('R-C: {}'.format(recall.mean()))
	print('')

	print('>>>>>>>>>>>>>>>>>>>>>>>> Overall Sample-Label Pairs >>>>>>>>>>>>>>>>>>>>>>>>>>>')
	precision, recall, accuracy, f1 = calc_acc_pr_f1_overall(gtruth.cpu().numpy().flatten(),
		preds.flatten())
	# macc = calc_mean_acc_overall(gtruth.cpu().numpy().flatten(), preds.flatten())
	# print('mA: {}'.format(macc) )
	print('F1_O: {}'.format(f1))
	print('P_O: {}'.format(precision))
	print('R_O: {}'.format(recall))


	print('mean mA')
	print(macc.mean())


if __name__ == '__main__':
	dataDir = '/data/hguo/Datasets/MS-COCO'
	dataType = 'val2014'
	trainset, valset = getSubsets(dataDir)
	test_loader = torch.utils.data.DataLoader(valset,
		batch_size = 8,
		shuffle = True,
		num_workers = 4)

	model_file = '/data/hguo/models/AttrFlow/mscoco_flow_resnet101_256/model_resnet101_1.pth'
	print(model_file)
	model = torch.load(model_file)
	model.eval()
	import time
	start_time = time.clock()
	test(model, test_loader, 1)
	end_time = time.clock()
	print('Test time: ', end_time - start_time)
	print('\n\n')
