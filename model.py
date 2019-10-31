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
		self.linear = nn.Linear(1280*16*16, 6*4)

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
		return self.conv3(out)

		
		

class Model(nn.Module):

	def __init__(self):
		super(Model, self).__init__()

		resnet18 = models.resnet18()
		self.res18_conv = nn.Sequential(*list(resnet18.children())[:-3])
		self.FPN =  fpn.FPN101()
		self.FCN = torchfcn.models.FCN8s(n_class=3)
		self.C_SEG = nn.ModuleList([ComponentSeg(i) for i in [2, 2, 2, 2, 2, 4]])
		self.C_PRED = ComponentPred()

	def forward(self, inp):

		# Component pred
		inp1 = self.res18_conv(inp)
		rect = self.C_PRED(inp1)
		
		# Component segm
		inp2 = self.FPN(inp)[0]
		eyebrow1 = rect[:,0:4]
		eyebrow2 = rect[:,4:8]
		eye1 = rect[:,8:12]
		eye2 = rect[:,12:16]
		nose = rect[:,16:20]
		mouth = rect[;,20:24]

		indx = torch.tensor(range(rect.shape[0]), dtype=torch.float).view(-1,1)
		eyebrow1 = ops.roi_align(inp2, torch.cat([indx,eyebrow1], dim=1), [32,32], spatial_scale=128./512., sampling_ratio=-1)
		eyebrow2 = ops.roi_align(inp2, torch.cat([indx,eyebrow2], dim=1), [32,32], spatial_scale=128./512., sampling_ratio=-1)
		eye1 = ops.roi_align(inp2, torch.cat([indx,eye1], dim=1), [32,32], spatial_scale=128./512., sampling_ratio=-1)
		eye2 = ops.roi_align(inp2, torch.cat([indx,eye2], dim=1), [32,32], spatial_scale=128./512., sampling_ratio=-1)
		nose = ops.roi_align(inp2, torch.cat([indx,nose], dim=1), [32,32], spatial_scale=128./512., sampling_ratio=-1)
		mouth = ops.roi_align(inp2, torch.cat([indx,mouth], dim=1), [32,32], spatial_scale=128./512., sampling_ratio=-1)


		segm_result = []
		segm_result.append(self.C_SEG[0](eyebrow1))
		segm_result.append(self.C_SEG[1](eyebrow2))
		segm_result.append(self.C_SEG[2](eye1))
		segm_result.append(self.C_SEG[3](eye2))
		segm_result.append(self.C_SEG[4](nose))
		segm_result.append(self.C_SEG[5](mouth))


		# FCN
		fcn_result = self.FCN(inp)


		return [rect, segm_result, fcn_result]


#m = Model()
#r,s,f =m.forward(torch.tensor(warped.transpose(2,0,1), dtype=torch.float).view(1, 3, 512,512))
#print(f.shape)