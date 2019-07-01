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

from sklearn.metrics import average_precision_score
from wider import get_subsets, imshow

num_attr = 14

def calc_average_precision(y_true, y_score):
	aps = np.zeros(num_attr)
	for i in range(num_attr):
		true = y_true[i]
		score = y_score[i]

		non_index = np.where(true == 0)
		score = np.delete(score, non_index)
		true = np.delete(true, non_index)

		true[true == -1.] = 0

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

		true[true == -1.] = 0

		precision[i] = metrics.precision_score(true, pred)
		recall[i] = metrics.recall_score(true, pred)
		accuracy[i] = metrics.accuracy_score(true, pred)
		f1[i] = metrics.f1_score(true, pred)

	return precision, recall, accuracy, f1

def calc_mean_acc(y_true, y_pred):
	macc = np.zeros(num_attr)
	for i in range(num_attr):
		true = y_true[i]	# -1, 0, 1
		pred = y_pred[i]	# 0, 1

		true[true == -1.] = 0

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

	true[true == -1.] = 0

	precision = metrics.precision_score(true, pred)
	recall = metrics.recall_score(true, pred)
	accuracy = metrics.accuracy_score(true, pred)
	f1 = metrics.f1_score(true, pred)

	return precision, recall, accuracy, f1

def calc_mean_acc_overall(y_true, y_pred):

	true = y_true	# 0, 1
	pred = y_pred	# 0, 1

	true[true == -1.] = 0

	temp = true + pred
	tp = (temp[temp == 2]).size
	tn = (temp[temp == 0]).size
	p = (true[true == 1]).size
	n = (true[true == 0]).size
	macc = .5 * tp / (p) + .5 * tn / (n)

	return macc

def eval_example(y_true, y_pred):
	# example-based metrics
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

		if ui != 0:
			acc += 1.0 * ii / ui
		if fi != 0:
			prec += 1.0 * ii / fi
		if yi != 0:
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
		images = sample[0]	# test just large
		labels = sample[4]
		labels = labels.type(torch.FloatTensor)

		images = images.cuda()
		labels = labels.cuda()

		test_input = Variable(images)
		y, _ = model(test_input)

		probs = torch.cat((probs, y.data.transpose(1, 0)), 1)
		gtruth = torch.cat((gtruth, labels.transpose(1, 0)), 1)

	print('prediction finished ....')

	preds = np.zeros((probs.size(0), probs.size(1)))
	temp = probs.cpu().numpy()
	preds[temp > 0.] = 1

	import scipy.io
	import os
	if not os.path.isdir('./preds'):
		os.mkdir('./preds')
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

	print("AP: {}".format(aps.mean()))
	print('F1-C: {}'.format(f1.mean()))
	print('P-C: {}'.format(precision.mean()))
	print('R-C: {}'.format(recall.mean()))
	print('')


	print('>>>>>>>>>>>>>>>>>>>>>>>> Overall Sample-Label Pairs >>>>>>>>>>>>>>>>>>>>>>>>>>>')
	precision, recall, accuracy, f1 = calc_acc_pr_f1_overall(gtruth.cpu().numpy().flatten(),
		preds.flatten())

	print('F1_O: {}'.format(f1))
	print('P_O: {}'.format(precision))
	print('R_O: {}'.format(recall))
	print('\n')


	macc = calc_mean_acc(gtruth.cpu().numpy(), preds)
	print('mA scores')
	print(macc)
	print('mean mA')
	print(macc.mean())

	print('\n')

if __name__ == '__main__':
	anno_dir = '/path/to/wider_attribute_annotation'
	data_dir = '/path/to/Image'
	trainset, testset = get_subsets(anno_dir, data_dir)
	test_loader = torch.utils.data.DataLoader(testset,
		batch_size = 16,
		shuffle = False,
		num_workers = 4)

	# modify to test multiple checkpoints continuously
	for i in range(11, 12):
		model_file = '/path/to/model_resnet50_{}.pth'.format(i)
		model = torch.load(model_file)
		print(model_file)
		model.eval()
		start_time = time.clock()
		test(model, test_loader, i)
		end_time = time.clock()
		print('Time: ', end_time - start_time)
		print('\n')
