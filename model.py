import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import torchvision.ops as ops
import torchfcn
from test import warped
import fpn
from torch.jit.annotations import List

class ComponentPred(nn.Module):
	"""docstring for ComponentPred"""
	def __init__(self):
		super(ComponentPred, self).__init__()
		self.conv1 = nn.Conv2d(256, 320, 3, padding=1)
		self.bnorm1 = nn.BatchNorm2d(320)
		self.conv2 = nn.Conv2d(320, 1280, 1)
		self.bnorm2 = nn.BatchNorm2d(1280)
		self.linear = nn.Linear(1280*16*16, 4*4)

	def forward(self, inp):
		b,c,h,w = inp.shape
		out = self.bnorm1(self.conv1(inp))
		return self.linear(F.avg_pool2d(self.bnorm2(self.conv2(out)), kernel_size=2, stride=2).view(b,-1))


		


class ComponentSeg(nn.Module):
	"""docstring for CompponentSeg"""
	def __init__(self, out_planes):
		super(ComponentSeg, self).__init__()

		self.conv1 = nn.Conv2d(256, 256, 3, padding=1)
		self.bnorm1 = nn.BatchNorm2d(256)
		self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
		self.bnorm2 = nn.BatchNorm2d(256)

		self.conv3 = nn.Conv2d(256, out_planes, 1)

	def forward(self, inp):

		out = F.interpolate(self.bnorm1(self.conv1(inp)), scale_factor=2, mode='bilinear')
		out = F.interpolate(self.bnorm2(self.conv2(out)), scale_factor=2, mode='bilinear')
		return F.softmax(self.conv3(out), dim=1)



		
		

class Model(nn.Module):

	def __init__(self):
		super(Model, self).__init__()

		resnet18 = models.resnet18()
		self.res18_conv = nn.Sequential(*list(resnet18.children())[:-3])
		self.FPN =  fpn.FPN101()
		self.FCN = torchfcn.models.FCN8s(n_class=3)
		self.C_SEG = nn.ModuleList([ComponentSeg(i) for i in [2, 2, 2, 4]])
		self.C_PRED = ComponentPred()

	def forward(self, inp):

		# Component pred
		inp1 = self.res18_conv(inp)
		rect = self.C_PRED(inp1)
		
		# Component segm
		inp2 = self.FPN(inp)
		lst = []
		for i in range(rect.shape[0]):
			lst.append(rect[i].view(-1,4))


		roi_features = ops.roi_align(inp2, , [32,32], spatial_scale=128./512., sampling_ratio=-1)
		print('features: ', roi_features.shape)

		segm_result = []
		for i in range(4):
			segm_result.append(roi_features[:,i,:,:,:])


		# FCN
		fcn_result = self.FCN(inp)


		return [rect, segm_result, fcn_result]


m = Model()

m.forward(torch.tensor(warped.transpose(2,0,1), dtype=torch.float).view(1, 3, 512,512))