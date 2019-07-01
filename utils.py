import torch
from wider import get_subsets
from torch.autograd import Variable


def get_dataset(opts):
	if opts["dataset"] == "WIDER":
		data_dir = opts["data_dir"]
		anno_dir = opts["anno_dir"]
		trainset, testset = get_subsets(anno_dir, data_dir)
	else:
		# will be added later
		pass

	return trainset, testset


def adjust_learning_rate(optimizer, epoch, args):
	"""Sets the learning rate to the initial LR decayed every 30 epochs"""
	lr = args.learning_rate * (args.decay ** (epoch // args.stepsize))
	print("Current learning rate is: {:.5f}".format(lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def SigmoidCrossEntropyLoss(x, y, w_p, w_n):
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