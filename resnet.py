import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo

from torch.autograd import Variable


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
		   'resnet152']


model_urls = {
	'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
	'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
	'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

num_attr = 14


def conv3x3(in_planes, out_planes, stride=1):
	"3x3 convolution with padding"
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
							   padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class ResNet(nn.Module):

	def __init__(self, block, layers, num_classes=1000):
		self.inplanes = 64
		super(ResNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AdaptiveAvgPool2d(1)

		self.fc_all = nn.Linear(512 * block.expansion, num_attr)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		# modify the forward function
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		feat = x
		N, C, H, W = feat.shape
		# global
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		y = self.fc_all(x)

		# local
		# get the FC parameters
		params = list(self.parameters())
		fc_weights = params[-2].data
		fc_weights = fc_weights.view(1, num_attr, C, 1, 1)
		fc_weights = Variable(fc_weights, requires_grad = False)

		# attention
		feat = feat.unsqueeze(1)	# N * 1 * C * H * W
		hm = feat * fc_weights
		hm = hm.sum(2)		# N * num_attr * H * W

		heatmap = hm

		return y, heatmap


def resnet18(pretrained=False, **kwargs):
	"""Constructs a ResNet-18 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
	if pretrained:
		pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
		model_dict = model.state_dict()
		pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)
		# model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
	return model


def resnet34(pretrained=False, **kwargs):
	"""Constructs a ResNet-34 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
	return model


def resnet50(pretrained=False, **kwargs):
	"""Constructs a ResNet-50 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
	if pretrained:
		pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
		model_dict = model.state_dict()
		# for k,v in pretrained_dict.items():
		# 	if k in model_dict:
		# 		print(k)
		pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)
		# model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
	return model


def resnet101(pretrained=False, **kwargs):
	"""Constructs a ResNet-101 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
	if pretrained:
		pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
		model_dict = model.state_dict()
		pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)
		# model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
	return model


def resnet152(pretrained=False, **kwargs):
	"""Constructs a ResNet-152 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
	return model
