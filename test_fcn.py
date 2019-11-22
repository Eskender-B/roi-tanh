import torchfcn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from preprocess import ToTensor, ImageDataset, Warp
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import argparse
from utils import LOG_INFO
import pickle
import matplotlib.pyplot as plt
from fcn import FCN8s, VGGNet




parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate for optimizer")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs to train")
parser.add_argument("--load_model", default=False, type=bool, help="Wether to continue training from last checkpoint")
parser.add_argument("--test_only", default=False, type=bool, help="Only testing")
args = parser.parse_args()
print(args)

if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")



train_dataset = ImageDataset(txt_file='exemplars.txt',
                                           root_dir='data/SmithCVPR2013_dataset_warped',
                                           bg_indexs=set([0,1,2,3,4,5,7,8,9,10]),
                                           transform=transforms.Compose([ToTensor()]))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)


valid_dataset = ImageDataset(txt_file='tuning.txt',
                                           root_dir='data/SmithCVPR2013_dataset_warped',
                                           bg_indexs=set([0,1,2,3,4,5,7,8,9,10]),
                                           transform=transforms.Compose([ToTensor()]))
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)


test_dataset = ImageDataset(txt_file='testing.txt',
                                           root_dir='data/SmithCVPR2013_dataset_resized',
                                           warp_on_fly=True,
                                           bg_indexs=set([0,1,2,3,4,5,7,8,9,10]),
                                           transform=transforms.Compose([ToTensor()]))

test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)



vgg_model = VGGNet(requires_grad=True)
model = FCN8s(pretrained_net=vgg_model, n_class=2)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
model = model.to(device)


criterion3 = nn.CrossEntropyLoss().to(device)



def train2(epoch, model, train_loader, optimizer):
	loss_list3 = []

	model.train()

	for j, batch in enumerate(train_loader):
		optimizer.zero_grad()
		image, labels, rects = batch['image'].to(device), batch['labels'].to(device), batch['rects'].to(device)
		
		full = model(F.interpolate(image, size=[128,128], mode='bilinear'))
		#full = model(image)
		
		## Loss3
		full_l = F.interpolate(labels.float(), size=[128,128], mode='bilinear').argmax(dim=1)
		#full_l = labels.argmax(dim=1)
		
		loss3 = criterion3(full, full_l)


		## Total loss
		loss3.backward()
		optimizer.step()


		loss_list3.append(loss3.item())

		if j % args.display_freq == 0:
			msg = "Epoch %02d, Iter [%03d/%03d], train loss = %.4f" % (
			epoch, j, len(train_loader), np.mean(loss_list3))
			LOG_INFO(msg)

			loss_list3.clear()




def evaluate2(model, loader):

	epoch_loss3 = 0
	model.eval()

	with torch.no_grad():
		for batch in loader:
				
			image, labels, rects = batch['image'].to(device), batch['labels'].to(device), batch['rects'].to(device)
			full = model(F.interpolate(image, size=[128,128], mode='bilinear'))
			#full = model(image)

			## Loss3
			full_l = F.interpolate(labels.float(), size=[128,128], mode='bilinear').argmax(dim=1)
			#full_l = labels.argmax(dim=1)
			
			loss3 = criterion3(full, full_l)

			epoch_loss3 += loss3.item()


	return epoch_loss3/len(loader)



def save_result(image, label, pred_label, idx):

	image = image * 255
	colors = np.array([[0,0,0], [255,0,0]])

	orig_mask = np.expand_dims(label,-1) * colors.reshape(2,1,1,3)
	orig_mask = orig_mask.sum(0).astype(np.float) # May need to fix here

	pred_mask = np.expand_dims(pred_label,-1) * colors.reshape(2,1,1,3)
	pred_mask = pred_mask.sum(0).astype(np.float) # May need to fix here

	alpha = 0.5
	ground = np.where(orig_mask==np.array([0., 0., 0.]), image, alpha*image + (1.-alpha)*orig_mask)
	pred = np.where(pred_mask==np.array([0., 0., 0.]), image, alpha*image + (1.-alpha)*pred_mask)  


	ground = np.uint8(ground.clip(0., 255.))
	pred = np.uint8(pred.clip(0.,255.))

	plt.figure(figsize=(12.8, 9.6))

	ax = plt.subplot(1, 2, 1)
	ax.set_title("Ground Truth")
	plt.imshow(ground)

	ax = plt.subplot(1, 2, 2)
	ax.set_title("Predicted")
	plt.imshow(pred)

	plt.savefig('res/'+test_dataset.name_list[idx, 1].strip() + '.jpg')
	print('Save: ', test_dataset.name_list[idx, 1].strip() + '.jpg')
	plt.close()



def score_save(images, labels, pred_labels, landmarks, orig_size, indexs):

	images = images.to('cpu').numpy().transpose(0,2,3,1)
	labels = labels.float().to('cpu').numpy().transpose(0,2,3,1)
	pred_labels = pred_labels.float().to('cpu').numpy().transpose(0,2,3,1)
	b,h,w,l = labels.shape
	orig_size=orig_size.numpy()

	for i in range(b):
		warp_obj = Warp(landmarks[i])
		dimage = warp_obj.inverse(images[i], orig_size[i])
		dlabel = F.one_hot(torch.from_numpy(warp_obj.inverse(labels[i], orig_size[i]).transpose(2,0,1)).argmax(dim=0), l).numpy().transpose(2,0,1)
		dpred_label = F.one_hot(torch.from_numpy(warp_obj.inverse(pred_labels[i], orig_size[i]).transpose(2,0,1)).argmax(dim=0), l).numpy().transpose(2,0,1)
		

		save_result(dimage, dlabel, dpred_label, indexs[i])


def test(model, loader):
	epoch_loss = 0
	model.eval()

	with torch.no_grad():
		for batch in loader:
				
			images, labels, rects, landmarks, indexs, orig_size = batch['image'].to(device), batch['labels'].to(device), batch['rects'].to(device), batch['landmarks'], batch['index'], batch['orig_size']
			full = model(F.interpolate(images, size=[128,128], mode='bilinear'))
			#full = model(images)

			res = F.one_hot(full.argmax(dim=1), 2).transpose(3,1).transpose(2,3)
			pred_labels = F.one_hot(F.interpolate(res.float(), size=[512,512], mode='bilinear').argmax(dim=1), 3).transpose(3,1).transpose(2,3)
			
			#pred_labels = F.one_hot(full.argmax(dim=1), 3).transpose(3,1).transpose(2,3)

			
			score_save(images, labels, pred_labels, landmarks, orig_size, indexs)



if args.test_only:
	model = pickle.load(open('res/saved-fcn.pth', 'rb'))
	test(model, test_loader)
else:
	LOSS = 100
	if args.load_model:
		model = pickle.load(open('res/saved-fcn.pth', 'rb'))

	for epoch in range(1, args.epochs + 1):
		train2(epoch, model, train_loader, optimizer)
		l3 = evaluate2(model, valid_loader)
		if l3 < LOSS:
			LOSS = l3
			epoch_min = epoch
			pickle.dump(model, open('res/saved-fcn.pth', 'wb'))

		msg = '...Epoch %02d, val loss = %.4f ' % (epoch, l3)
		LOG_INFO(msg)


	model = pickle.load(open('res/saved-fcn.pth', 'rb'))
	msg = 'Min @ Epoch %02d, val loss = %.4f' % (epoch_min, LOSS)
	LOG_INFO(msg)
	l3 = evaluate2(model, test_loader)
	LOG_INFO('Finally, test loss = %.4f' % l3)
	test(model, test_loader)