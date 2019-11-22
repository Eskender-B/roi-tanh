import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import torchvision.ops as ops
import torchfcn
import fpn
from torch.jit.annotations import List
from fcn import FCN8s, VGGNet

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
		
		self.vgg_model = VGGNet(requires_grad=True)
		self.FCN = FCN8s(pretrained_net=self.vgg_model, n_class=3)
		self.fcn_conv = nn.Conv2d(256, 3, 1)
		self.fcn_bnorm = nn.BatchNorm2d(3)
		
		self.C_SEG = nn.ModuleList([ComponentSeg(i) for i in [2, 2, 2, 2, 2, 4]])
		self.C_PRED = ComponentPred()

	def forward(self, inp, rects_only=False):

		# Component pred
		inp1 = self.res18_conv.forward(inp)
		rect = self.C_PRED.forward(inp1)

		if rects_only:
			return rect
		
		# Component segm
		inp2 = self.FPN.forward(inp)[0]
		eyebrow1 = rect[:,0:4]
		eyebrow2 = rect[:,4:8]
		eye1 = rect[:,8:12]
		eye2 = rect[:,12:16]
		nose = rect[:,16:20]
		mouth = rect[:,20:24]

		indx = torch.tensor(range(rect.shape[0]), dtype=torch.float).view(-1,1).to(inp.device)
		eyebrow1 = ops.roi_align(inp2, torch.cat([indx,eyebrow1], dim=1), [32,32], spatial_scale=128./512., sampling_ratio=-1)
		eyebrow2 = ops.roi_align(inp2, torch.cat([indx,eyebrow2], dim=1), [32,32], spatial_scale=128./512., sampling_ratio=-1)
		eye1 = ops.roi_align(inp2, torch.cat([indx,eye1], dim=1), [32,32], spatial_scale=128./512., sampling_ratio=-1)
		eye2 = ops.roi_align(inp2, torch.cat([indx,eye2], dim=1), [32,32], spatial_scale=128./512., sampling_ratio=-1)
		nose = ops.roi_align(inp2, torch.cat([indx,nose], dim=1), [32,32], spatial_scale=128./512., sampling_ratio=-1)
		mouth = ops.roi_align(inp2, torch.cat([indx,mouth], dim=1), [32,32], spatial_scale=128./512., sampling_ratio=-1)


		segm_result = []
		segm_result.append(self.C_SEG[0].forward(eyebrow1))
		segm_result.append(self.C_SEG[1].forward(eyebrow2))
		segm_result.append(self.C_SEG[2].forward(eye1))
		segm_result.append(self.C_SEG[3].forward(eye2))
		segm_result.append(self.C_SEG[4].forward(nose))
		segm_result.append(self.C_SEG[5].forward(mouth))


		# FCN
		#fcn_result = self.FCN.forward(inp2)
		fcn_result = self.FCN.forward( self.fcn_bnorm(self.fcn_conv(inp2)) )
		#fcn_result = self.FCN.forward( F.interpolate(inp, size=[128,128], mode='bilinear') )



		return [rect, segm_result, fcn_result]


#m = Model()
#r,s,f =m.forward(torch.tensor(warped.transpose(2,0,1), dtype=torch.float).view(1, 3, 512,512))
#print(f.shape)