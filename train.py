import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from preprocess import ToTensor, ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import argparse
from utils import LOG_INFO
import pickle
from model import Model




parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate for optimizer")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs to train")
args = parser.parse_args()
print(args)

if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")



train_dataset = ImageDataset(txt_file='exemplars.txt',
                                           root_dir='data/SmithCVPR2013_dataset_warped',
                                           bg_indexs=set([0]),
                                           transform=transforms.Compose([ToTensor()]))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)


valid_dataset = ImageDataset(txt_file='tuning.txt',
                                           root_dir='data/SmithCVPR2013_dataset_warped',
                                           bg_indexs=set([0]),
                                           transform=transforms.Compose([ToTensor()]))
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)


test_dataset = ImageDataset(txt_file='testing.txt',
                                           root_dir='data/SmithCVPR2013_dataset_warped',
                                           bg_indexs=set([0]),
                                           transform=transforms.Compose([ToTensor()]))

test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)



model = Model()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
model = model.to(device)

criterion1 = nn.L1Loss().to(device)

criterion2 = []
for i in range(6):
	criterion2.append(nn.CrossEntropyLoss().to(device))

criterion3 = nn.CrossEntropyLoss().to(device)




def train1(epoch, model, train_loader, optimizer):
	loss_list = []
	model.train()

	for i, batch in enumerate(train_loader):
		optimizer.zero_grad()
		image, rects = batch['image'].to(device), batch['rects'].to(device)
		pred_rects = model(image,rects_only=True)
		loss = criterion1(pred_rects, rects)

		loss.backward()
		optimizer.step()


		loss_list.append(loss.item())


		if i % args.display_freq == 0:
			msg = "Epoch %02d, Iter [%03d/%03d], train loss = %.4f" % (
			epoch, i, len(train_loader), np.mean(loss_list))
			LOG_INFO(msg)
			loss_list.clear()


def train2(epoch, model, train_loader, optimizer):
	loss_list = []
	loss_list1 = []
	loss_list2 = []
	loss_list3 = []

	model.train()

	for j, batch in enumerate(train_loader):
		optimizer.zero_grad()
		image, labels, rects = batch['image'].to(device), batch['labels'].to(device), batch['rects'].to(device)
		pred_rects, segm, full = model(image)

		## Loss1
		loss1 = criterion1(pred_rects, rects)


		## Loss2
		parts = [torch.tensor([]).to(device).long()] * 6
		b,c,h,w = image.shape

		rects_ = pred_rects.round().long().to('cpu').clamp(0,511)
		for bb in range(b):
			# Non-mouth
			for i in range(5):
				x1, y1, x2, y2 = rects_[bb][i*4:(i+1)*4]
				if x2-x1 <= 0 or y2-y1<=0:
					x1, y1, x2, y2 = torch.tensor([0,0,511,511]).long()

				p = torch.gather(labels[bb][2+i], 0,  torch.arange(y1, y2+1).unsqueeze(1).repeat_interleave(w,dim=1).to(device))
				p = torch.gather(p, 1,  torch.arange(x1, x2+1).unsqueeze(0).repeat_interleave(y2-y1+1,dim=0).to(device))
				p = torch.stack([1-p, p])  # background
				parts[i] = torch.cat([parts[i], F.interpolate(p.float().unsqueeze(0), size=[128,128], mode='bilinear').squeeze(0).argmax(dim=0)], dim=0)

			# Mouth
			x1, y1, x2, y2 = rects_[bb][20:24]
			if x2-x1 <= 0 or y2-y1<=0:
				x1, y1, x2, y2 = torch.tensor([0,0,511,511]).long()

			p = torch.gather(labels[bb][7:10], 1,  torch.arange(y1, y2+1).unsqueeze(1).repeat_interleave(w,dim=1).unsqueeze(0).repeat_interleave(3,dim=0).to(device))
			p = torch.gather(p, 2,  torch.arange(x1, x2+1).unsqueeze(0).repeat_interleave(y2-y1+1,dim=0).unsqueeze(0).repeat_interleave(3,dim=0).to(device))
			p = torch.cat([1-p.sum(dim=0,keepdim=True), p], 0)  # background
			parts[5] = torch.cat([parts[5], F.interpolate(p.float().unsqueeze(0), size=[128,128], mode='bilinear').squeeze(0).argmax(dim=0)], dim=0)



		loss2 = []
		for i in range(6):
			loss2.append(criterion2[i](segm[i], parts[i].view(b,128,128)))

		## Loss3
		full_l = F.interpolate(labels.float().index_select(1, torch.tensor([0,1,10]).long().to(device)), size=[128,128], mode='bilinear').argmax(dim=1)
		loss3 = criterion3(full, full_l)


		## Total loss
		tot_loss = loss1 + sum(loss2) + loss3
		tot_loss.backward()
		optimizer.step()


		loss_list1.append(loss1.item())
		loss_list2.append((sum(loss2)/6.).item())
		loss_list3.append(loss3.item())
		loss_list.append(tot_loss.item())
		if j % args.display_freq == 0:
			msg = "Epoch %02d, Iter [%03d/%03d], train loss = %.4f, %.4f, %.4f, %.4f" % (
			epoch, j, len(train_loader), np.mean(loss_list1),np.mean(loss_list2),np.mean(loss_list3),np.mean(loss_list))
			LOG_INFO(msg)
			loss_list.clear()
			loss_list1.clear()
			loss_list2.clear()
			loss_list3.clear()



def evaluate1(model, loader):
	epoch_loss = 0
	model.eval()

	with torch.no_grad():
		for batch in loader:
			image, rects = batch['image'].to(device), batch['rects'].to(device)
			pred_rects = model(image,rects_only=True)
			loss = criterion1(pred_rects, rects)

			epoch_loss += loss.item()

	return epoch_loss / len(loader)


def evaluate2(model, loader):
	epoch_loss1 = 0
	epoch_loss2 = 0
	epoch_loss3 = 0
	epoch_tot = 0
	model.eval()

	with torch.no_grad():
		for batch in loader:
				
			image, labels, rects = batch['image'].to(device), batch['labels'].to(device), batch['rects'].to(device)
			pred_rects, segm, full = model(image)

			## Loss1
			loss1 = criterion1(pred_rects, rects)


			## Loss2
			parts = [torch.tensor([]).to(device).long()] * 6
			b,c,h,w = image.shape

			rects_ = pred_rects.round().long().to('cpu').clamp(0,511)
			for bb in range(b):
				# Non-mouth
				for i in range(5):
					x1, y1, x2, y2 = rects_[bb][i*4:(i+1)*4]
					if x2-x1 <= 0 or y2-y1<=0:
						x1, y1, x2, y2 = torch.tensor([0,0,511,511]).long()

					p = torch.gather(labels[bb][2+i], 0,  torch.arange(y1, y2+1).unsqueeze(1).repeat_interleave(w,dim=1).to(device))
					p = torch.gather(p, 1,  torch.arange(x1, x2+1).unsqueeze(0).repeat_interleave(y2-y1+1,dim=0).to(device))
					p = torch.stack([1-p, p])  # background
					parts[i] = torch.cat([parts[i], F.interpolate(p.float().unsqueeze(0), size=[128,128], mode='bilinear').squeeze(0).argmax(dim=0)], dim=0)

				# Mouth
				x1, y1, x2, y2 = rects_[bb][20:24]
				if x2-x1 <= 0 or y2-y1<=0:
					x1, y1, x2, y2 = torch.tensor([0,0,511,511]).long()
			
				p = torch.gather(labels[bb][7:10], 1,  torch.arange(y1, y2+1).unsqueeze(1).repeat_interleave(w,dim=1).unsqueeze(0).repeat_interleave(3,dim=0).to(device))
				p = torch.gather(p, 2,  torch.arange(x1, x2+1).unsqueeze(0).repeat_interleave(y2-y1+1,dim=0).unsqueeze(0).repeat_interleave(3,dim=0).to(device))
				p = torch.cat([1-p.sum(dim=0,keepdim=True), p], 0)  # background
				parts[5] = torch.cat([parts[5], F.interpolate(p.float().unsqueeze(0), size=[128,128], mode='bilinear').squeeze(0).argmax(dim=0)], dim=0)



			loss2 = []
			for i in range(6):
				loss2.append(criterion2[i](segm[i], parts[i].view(b,128,128)))

			## Loss3
			full_l = F.interpolate(labels.float().index_select(1, torch.tensor([0,1,10]).long().to(device)), size=[128,128], mode='bilinear').argmax(dim=1)
			loss3 = criterion3(full, full_l)


			## Total loss
			tot_loss = loss1 + sum(loss2) + loss3


			epoch_loss1 += loss1.item()
			epoch_loss2 += (sum(loss2)/6).item()
			epoch_loss3 += loss3.item()
			epoch_tot += tot_loss.item()

	return [epoch_loss1/len(loader), epoch_loss2/len(loader), epoch_loss3/len(loader), epoch_tot/len(loader)]


LOSS = 100
epoch_min = 1

for epoch in range(1, args.epochs + 1):
	train1(epoch, model, train_loader, optimizer)
	valid_loss = evaluate1(model, valid_loader)
	msg = '...Epoch %02d, val loss = %.4f' % (epoch, valid_loss)
	LOG_INFO(msg)


for epoch in range(1, args.epochs + 1):
	train2(epoch, model, train_loader, optimizer)
	l1,l2,l3,lt = evaluate2(model, valid_loader)
	if lt < LOSS:
		LOSS = lt
		epoch_min = epoch
		pickle.dump(model, open('res/saved-model.pth', 'wb'))

	msg = '...Epoch %02d, val loss = %.4f, %.4f, %.4f, %.4f' % (epoch, l1,l2,l3,lt)
	LOG_INFO(msg)


model = pickle.load(open('res/saved-model.pth', 'rb'))
msg = 'Min @ Epoch %02d, val loss = %.4f' % (epoch_min, LOSS)
LOG_INFO(msg)
l1,l2,l3,lt = evaluate2(model, test_loader)
LOG_INFO('Finally, test loss = %.4f, %.4f, %.4f, %.4f' % (l1,l2,l3,lt))