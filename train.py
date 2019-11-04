import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from preprocess import ToTensor, ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils




parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate for optimizer")
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
                                           transform=transforms.Compose([                                               ,
                                               ToTensor()
                                           ]))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)


valid_dataset = ImageDataset(txt_file='tuning.txt',
                                           root_dir='data/SmithCVPR2013_dataset_warped',
                                           bg_indexs=set([0]),
                                           transform=transforms.Compose([
                                               ToTensor()
                                           ]))
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)


test_dataset = ImageDataset(txt_file='testing.txt',
                                           root_dir='data/SmithCVPR2013_dataset_warped',
                                           bg_indexs=set([0]),
                                           transform=transforms.Compose([
                                               ToTensor(),
                                           ]))
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)



model = Model()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
model = model.to(device)

criterion1 = nn.L1Loss().to(device)

criterion2 = []
for i in range(6):
	criterion2.append(nn.CrossEntropyLoss().to(device))

criterion3 = nn.CrossEntropyLoss().to(device)




def train1(epoch, model, train_loader, optimizer, criterion):
	loss_list = []
	model.train()

	for i, batch in enumerate(train_loader):
		optimizer.zero_grad()
		image, rects = batch['image'].to(device), batch['rects'].to(device)
		pred_rects, segm, full = model(image)
		loss = criterion1(pred_rects, rects)

		loss.backward()
		optimizer.step()


		loss_list.append(loss.item())


		if i % args.display_freq == 0:
			msg = "Epoch %02d, Iter [%03d/%03d], train loss = %.4f" % (
			epoch, i, len(train_loader), np.mean(loss_list))
			LOG_INFO(msg)
			loss_list.clear()


def train2(epoch, model, train_loader, optimizer, criterion):
	loss_list = []
	model.train()

	for i, batch in enumerate(train_loader):
		optimizer.zero_grad()
		image, labels, rects = batch['image'].to(device), batch['labels'].to(device), batch['rects'].to(device)
		pred_rects, segm, full = model(image)

		## Loss1
		loss1 = criterion1(pred_rects, rects)


		## Loss2
		parts = [torch.tensor([])] * 6
		b,c,h,w = image.shape

		for bb in range(b):
			# Non-mouth
			for i in range(5):
				x1, y1, x2, y2 = rects[bb][i*4:(i+1)*4].round()
				p = torch.gather(labels[bb][2+i], 0,  torch.arange(y1, y2+1).unsqueeze(1).repeat_interleave(w,dim=1))
				p = torch.gather(p, 1,  torch.arange(x1, x2+1).unsqueeze(0).repeat_interleave(y2-y1+1,dim=0))
				p = torch.stack([1-p, p])  # background
				parts[i] = torch.cat([parts[i], F.interpolate(p, [128,128], mode='linear').argmax(dim=0)], dim=0)

			# Mouth
			x1, y1, x2, y2 = rects[bb][20:24].round()
			p = torch.gather(labels[bb][7:10], 0,  torch.arange(y1, y2+1).unsqueeze(1).repeat_interleave(w,dim=1).unsqueeze(0).repeat_interleave(3,dim=0))
			p = torch.gather(p, 1,  torch.arange(x1, x2+1).unsqueeze(0).repeat_interleave(y2-y1+1,dim=0).unsqueeze(0).repeat_interleave(3,dim=0))
			p = torch.cat([1-p.sum(dim=0,keepdim=True), p], 0)  # background
			parts[i] = torch.cat([parts[i], F.one_hot(F.interpolate(p, [128,128], mode='linear').argmax(dim=0), 4).transpose(0,2).transpose(1,2)], dim=0)



		loss2 = []
		for i in range(6):
			loss2.append(criterion2(segm[i], parts[i].view(b,128,128)))

		## Loss3
		loss3 = criterion3(full, torch.index_select(labels, dim=1, torch.tensor([0,1,10]).long()).argmax(dim=1, keepdim=False))


		## Total loss
		tot_loss = loss1 + sum(loss2) + loss3
		tot_loss.backward()
		optimizer.step()


		loss_list.append(tot_loss.item())
		if i % args.display_freq == 0:
			msg = "Epoch %02d, Iter [%03d/%03d], train loss = %.4f" % (
			epoch, i, len(train_loader), np.mean(loss_list))
			LOG_INFO(msg)
			loss_list.clear()



def evaluate1(model, loader, criterion):
	epoch_loss = 0
	model.eval()

	with torch.no_grad():
		for batch in loader:
			image, rects = batch['image'].to(device), batch['rect'].to(device)
			rects_pred, segm, full = model(image)
			loss = criterion1(rects_pred, rects)

			epoch_loss += loss.item()

	return epoch_loss / len(loader)


def evaluate2(model, loader, criterion):
	epoch_loss = 0
	model.eval()

	with torch.no_grad():
		for batch in loader:
				
			image, labels, rects = batch['image'].to(device), batch['labels'].to(device), batch['rects'].to(device)
			pred_rects, segm, full = model(image)

			## Loss1
			loss1 = criterion1(pred_rects, rects)


			## Loss2
			parts = [torch.tensor([])] * 6
			b,c,h,w = image.shape

			for bb in range(b):
				# Non-mouth
				for i in range(5):
					x1, y1, x2, y2 = rects[bb][i*4:(i+1)*4].round()
					p = torch.gather(labels[bb][2+i], 0,  torch.arange(y1, y2+1).unsqueeze(1).repeat_interleave(w,dim=1))
					p = torch.gather(p, 1,  torch.arange(x1, x2+1).unsqueeze(0).repeat_interleave(y2-y1+1,dim=0))
					p = torch.stack([1-p, p])  # background
					parts[i] = torch.cat([parts[i], F.interpolate(p, [128,128], mode='linear').argmax(dim=0)], dim=0)

				# Mouth
				x1, y1, x2, y2 = rects[bb][20:24].round()
				p = torch.gather(labels[bb][7:10], 0,  torch.arange(y1, y2+1).unsqueeze(1).repeat_interleave(w,dim=1).unsqueeze(0).repeat_interleave(3,dim=0))
				p = torch.gather(p, 1,  torch.arange(x1, x2+1).unsqueeze(0).repeat_interleave(y2-y1+1,dim=0).unsqueeze(0).repeat_interleave(3,dim=0))
				p = torch.cat([1-p.sum(dim=0,keepdim=True), p], 0)  # background
				parts[i] = torch.cat([parts[i], F.one_hot(F.interpolate(p, [128,128], mode='linear').argmax(dim=0), 4).transpose(0,2).transpose(1,2)], dim=0)



			loss2 = []
			for i in range(6):
				loss2.append(criterion2(segm[i], parts[i].view(b,128,128)))

			## Loss3
			loss3 = criterion3(full, torch.index_select(labels, dim=1, torch.tensor([0,1,10]).long()).argmax(dim=1, keepdim=False))


			## Total loss
			tot_loss = loss1 + sum(loss2) + loss3

			tot_loss = loss1 + sum(loss2) + loss3

			epoch_loss += tot_loss.item()

	return epoch_loss / len(loader)


LOSS = 100
epoch_min = 1


for epoch in range(1, args.epochs + 1):
	train1(epoch, model, train_loader, optimizer, criterion)
	valid_loss = evaluate1(model, valid_loader, criterion)
	msg = '...Epoch %02d, val loss = %.4f' % (epoch, valid_loss)
	LOG_INFO(msg)




for epoch in range(1, args.epochs + 1):
	train2(epoch, model, train_loader, optimizer, criterion)
	valid_loss = evaluate2(model, valid_loader, criterion)
	if valid_loss < LOSS:
		LOSS = valid_loss
		epoch_min = epoch
		pickle.dump(model, open('res/saved-model.pth', 'wb'))

	msg = '...Epoch %02d, val loss = %.4f' % (epoch, valid_loss)
	LOG_INFO(msg)


model = pickle.load(open('res/saved-model.pth', 'rb'))


msg = 'Min @ Epoch %02d, val loss = %.4f' % (epoch_min, LOSS)
LOG_INFO(msg)
test_loss = evaluate2(model, test_loader, criterion)
LOG_INFO('Finally, test loss = %.4f' % (test_loss))