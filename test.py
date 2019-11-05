import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from preprocess import ToTensor, ImageDataset, Warp
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import pickle
import argparse
from utils import LOG_INFO
import pickle
from model import Model



parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size to use during testing.")
args = parser.parse_args()
print(args)

if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")


test_dataset = ImageDataset(txt_file='testing.txt',
                                           root_dir='data/SmithCVPR2013_dataset_resized',
                                           bg_indexs=set([0]),
                                           warp_on_fly=True,
                                           transform=transforms.Compose([
                                               ToTensor(),
                                           ])
                                           )
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)



model = pickle.load(open('res/saved-model.pth', 'rb'))
criterion1 = nn.L1Loss().to(device)

def combine_results(rects, segm, full):
	
	batch_size,_,_,_ = full.shape
	pred_labels = torch.zeros([batch_size,11,512,512], dtype=torch.long)

	for bb in range(batch_size):
		# Non-mouth parts
		for i in range(5):
			x1, y1, x2, y2 = rects[bb][i*4:(i+1)*4].round()
			pred_labels[bb,i+2,y1:y2+1,x1:x2+1] =  F.interpolate(segm[i][bb], [y2-y1+1,x2-x1+1], mode='linear').argmax(dim=0)

		# Mouth parts
		x1, y1, x2, y2 = rects[bb][20:24].round()
		pred_labels[bb,7:10,y1:y2+1,x1:x2+1] =  F.one_hot(F.interpolate(segm[5][bb], [y2-y1+1,x2-x1+1], mode='linear').argmax(dim=0) , 4).transpose(2,0).transpose(1,2)[1:4]


	# background,skin,hair
	full = F.one_hot(full.argmax(dim=1), 3).transpose(3,1).transpose(2,3)
	pred_labels[:,0:2,:,:] =  full[:,0:2,:,:].argmax()
	pred_labels[:,10,:,:] =  full[:,2,:,:]

	return pred_labels


TP = {'skin':0, 'eyebrow1':0, 'eyebrow2':0, 'eye1':0, 'eye2':0, 'nose':0, 'u_lip':0, 'i_mouth':0, 'l_lip':0, 'hair':0}
FP = {'skin':0, 'eyebrow1':0, 'eyebrow2':0, 'eye1':0, 'eye2':0, 'nose':0, 'u_lip':0, 'i_mouth':0, 'l_lip':0, 'hair':0}
TN = {'skin':0, 'eyebrow1':0, 'eyebrow2':0, 'eye1':0, 'eye2':0, 'nose':0, 'u_lip':0, 'i_mouth':0, 'l_lip':0, 'hair':0}
FN = {'skin':0, 'eyebrow1':0, 'eyebrow2':0, 'eye1':0, 'eye2':0, 'nose':0, 'u_lip':0, 'i_mouth':0, 'l_lip':0, 'hair':0}


def calculate_F1(labels, pred_labels):
  global TP, FP, TN, FN
  for i,name in enumerate(['skin', 'eyebrow1', 'eyebrow2' 'eye1', 'eye2', 'nose', 'u_lip', 'i_mouth', 'l_lip', 'hair']):
    TP[name]+= (labels[:,i+1,:,:] * pred_labels[:,i+1,:,:]).sum().tolist()
    FP[name]+= ( ((labels[:,i+1,:,:]-1)*-1) * pred_labels[:,i+1,:,:]).sum().tolist()
    TN[name]+= ( ((labels[:,i+1,:,:]-1)*-1) * ((pred_labels[:,i+1,:,:]-1)*-1) ).sum().tolist()
    FN[name]+= (labels[:,i+1,:,:] * ((pred_labels[:,i+1,:,:]-1)*-1) ).sum().tolist()


def show_F1():
  F1 = {}
  PRECISION = {}
  RECALL = {}
  tot_p = 0.0
  tot_r = 0.0
  for key in TP:
    PRECISION[key] = float(TP[key]) / (TP[key] + FP[key])
    RECALL[key] = float(TP[key]) / (TP[key] + FN[key])
    F1[key] = 2.*PRECISION[key]*RECALL[key]/(PRECISION[key]+RECALL[key])

    tot_p += PRECISION[key]
    tot_r += RECALL[key]

  #avg_p = tot_p/len(TP)
  #avg_r = tot_r/len(TP)
  #overall_F1 = 2.* avg_p*avg_r/ (avg_p+avg_r)

  mouth_p = (PRECISION['u_lip'] + PRECISION['i_mouth'] + PRECISION['l_lip'])/3.0
  mouth_r = (RECALL['u_lip'] + RECALL['i_mouth'] + RECALL['l_lip'])/3.0
  mouth_F1 = 2.* mouth_p * mouth_r / (mouth_p+mouth_r)

  avg_p, avg_r = tot_p/len(PRECISION), tot_r/len(RECALL)

  overall_F1 = 2.* avg_p*avg_r/ (avg_p+avg_r)


  print("\n\n", "PART\t\t", "F1-MEASURE ", "PRECISION ", "RECALL")
  for k in F1:
    print("%s\t\t"%k, "%.4f\t"%F1[k], "%.4f\t"%PRECISION[k], "%.4f\t"%RECALL[k])

  print("mouth(all)\t", "%.4f\t"%mouth_F1, "%.4f\t"%mouth_p, "%.4f\t"%mouth_r)
  print("Overall\t\t", "%.4f\t"%overall_F1, "%.4f\t"%avg_p, "%.4f\t"%avg_r)



def save_results(images, labels, pred_labels, indexs):

	images = images*255
	colors = torch.tensor([[0,0,0], [255,255,0], [255,0,0], [255,0,0], [0,0,255], [0,0,255], [255,165,0], [0,255,255], [0,255,0], [255,0,255], [150,75,0]]).to(device)

	orig_mask = labels.unsqueeze(-1) * colors.view(1,11,1,1,3)
	orig_mask = orig_mask.sum(1).float() # May need to fix here

	pred_mask = pred_labels.unsqueeze(-1) * colors.view(1,11,1,1,3)
	pred_mask = pred_mask.sum(1).float() # May need to fix here

	alpha = 0.1
	ground = torch.where(orig_mask==torch.tensor([0., 0., 0.]).to(device), images, alpha*images + (1.-alpha)*orig_mask)
	pred = torch.where(pred_mask==torch.tensor([0., 0., 0.]).to(device), images, alpha*images + (1.-alpha)*pred_mask)  


	ground = np.uint8(ground.clamp(0., 255.).to('cpu').numpy())
	pred = np.uint8(pred.detach().clamp(0.,255.).to('cpu').numpy())

	for i,idx in enumerate(indexs):
		plt.figure(figsize=(12.8, 9.6))

		ax = plt.subplot(1, 2, 1)
		ax.set_title("Ground Truth")
		plt.imshow(ground)

		ax = plt.subplot(1, 2, 2)
		ax.set_title("Predicted")
		plt.imshow(pred)

		plt.savefig('res/'+test_dataset.name_list[idx, 1].strip() + '.jpg')
		plt.close()


def dewarp(images, labels, pred_labels, landmarks, orig_size):
	images = images.numpy().transpose(0,2,3,1)
	labels = labels.float().numpy().transpose(0,2,3,1)
	pred_labels = pred_labels.float().numpy().transpose(0,2,3,1)
	b,h,w,l = labels.shape

	dimages, dlabels, dpred_labels = [],[],[]
	for i in range(b):
		warp_obj = Warp(landmarks[i])
		dimages.append(warp_obj.inverse(images[i], orig_size[i]))
		dlabels.append(warp_obj.inverse(labels[i], orig_size[i]))
		dpred_labels.append(warp_obj.inverse(pred_labels[i], orig_size[i]))

	dimages = torch.from_numpy(np.array(dimages).transpose(0,3,1,2))
	dlabels = F.one_hot( torch.from_numpy(np.array(dlabels).transpose(0,3,1,2)), l).transpose(1,3).transpose(2,3)
	dpred_labels = F.one_hot( torch.from_numpy(np.array(dpred_labels).transpose(0,3,1,2)), l).transpose(1,3).transpose(2,3)

	return dimages, dlabels, dpred_labels



r_error =0.0
def rects_error(rects, pred_rects):
	r_error += criterion1(rects, pred_rects).item()


def test(model, loader, criterion):
	epoch_loss = 0
	model.eval()

	with torch.no_grad():
		for batch in loader:
				
			images, labels, rects, landmarks, indexs, orig_size = batch['image'].to(device), batch['labels'].to(device), batch['rects'].to(device), batch['landmarks'], batch['indexs'], batch['orig_size']
			pred_rects, segm, full = model(images)

			pred_labels = combine_results(pred_rects, segm, full)
			images, labels, pred_labels = dewarp(images, labels, pred_labels, landmarks, orig_size)
			
			rects_error(rects, pred_rects)
			calculate_F1(labels, pred_labels)
			save_results(images, labels, pred_labels, indexs)



if __name__ == '__main__':
	test()
	show_F1()
	print()
	print("Rects Error: ", r_error/len(test_loader))