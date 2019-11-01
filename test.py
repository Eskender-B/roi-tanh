import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from preprocess import ToTensor, ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import pickle




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


test_dataset = ImageDataset(txt_file='testing.txt',
                                           root_dir='data/SmithCVPR2013_dataset_resized',
                                           bg_indexs=set([0]),
                                           transform=transforms.Compose([
                                               ToTensor(),
                                           ]))
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)



model = pickle.load(open('res/saved-model.pth', 'rb'))


def combine_results(rects, segm, full):
	
	batch_size,_,_,_ = full.shape
	pred_labels = torch.zeros([batch_size,11,512,512], dtype=torch.long)

	for bb in range(batch_size):
		for i in range(6):
			x1, y1, x2, y2 = rects[bb][i*4:(i+1)*4].round()
			pred_labels[bb,i+2,y1:y2+1,x1:x2+1] +=  F.interpolate(segm[i][bb].argmax(dim=1), [y2-y1+1,x2-x1+1], mode='bilinear')



    
  return ground_result, pred_result

def save_results(ground, pred, indexs, offsets, shapes):

  ground = np.uint8(ground.clamp(0., 255.).to('cpu').numpy())
  pred = np.uint8(pred.detach().clamp(0.,255.).to('cpu').numpy())

  for i,idx in enumerate(indexs):
    y1,x1 = offsets[i]
    y2,x2 = offsets[i] + shapes[i]
    plt.figure(figsize=(12.8, 9.6))

    ax = plt.subplot(1, 2, 1)
    ax.set_title("Ground Truth")
    plt.imshow(ground[i,y1:y2,x1:x2,:])

    ax = plt.subplot(1, 2, 2)
    ax.set_title("Predicted")
    plt.imshow(pred[i,y1:y2,x1:x2,:])

    plt.savefig('res/'+unresized_dataset.name_list[idx, 1].strip() + '.jpg')
    plt.close()


TP = {'eyebrow':0, 'eye':0, 'nose':0, 'u_lip':0, 'i_mouth':0, 'l_lip':0}
FP = {'eyebrow':0, 'eye':0, 'nose':0, 'u_lip':0, 'i_mouth':0, 'l_lip':0}
TN = {'eyebrow':0, 'eye':0, 'nose':0, 'u_lip':0, 'i_mouth':0, 'l_lip':0}
FN = {'eyebrow':0, 'eye':0, 'nose':0, 'u_lip':0, 'i_mouth':0, 'l_lip':0}

def calculate_F1(labels, pred_labels):
  global TP, FP, TN, FN
  for name in ['eyebrow', 'eye', 'nose']:
    TP[name]+= (batches[name]['labels'][:,0,:,:] * pred_labels[name][:,0,:,:]).sum().tolist()
    FP[name]+= (batches[name]['labels'][:,1,:,:] * pred_labels[name][:,0,:,:]).sum().tolist()
    TN[name]+= (batches[name]['labels'][:,1,:,:] * pred_labels[name][:,1,:,:]).sum().tolist()
    FN[name]+= (batches[name]['labels'][:,0,:,:] * pred_labels[name][:,1,:,:]).sum().tolist()

  ground = torch.cat( [batches['mouth']['labels'].index_select(1, torch.tensor([0]).to(device)), batches['mouth']['labels'].index_select(1, torch.tensor([1,2,3]).to(device)).sum(1, keepdim=True)], 1)
  pred = torch.cat( [pred_labels['mouth'].index_select(1, torch.tensor([0]).to(device)), pred_labels['mouth'].index_select(1, torch.tensor([1,2,3]).to(device)).sum(1, keepdim=True)], 1)
  TP['u_lip']+= (ground[:,0,:,:] * pred[:,0,:,:]).sum().tolist()
  FP['u_lip']+= (ground[:,1,:,:] * pred[:,0,:,:]).sum().tolist()
  TN['u_lip']+= (ground[:,1,:,:] * pred[:,1,:,:]).sum().tolist()
  FN['u_lip']+= (ground[:,0,:,:] * pred[:,1,:,:]).sum().tolist()

  ground = torch.cat( [batches['mouth']['labels'].index_select(1, torch.tensor([1]).to(device)), batches['mouth']['labels'].index_select(1, torch.tensor([0,2,3]).to(device)).sum(1, keepdim=True)], 1)
  pred = torch.cat( [pred_labels['mouth'].index_select(1, torch.tensor([1]).to(device)), pred_labels['mouth'].index_select(1, torch.tensor([0,2,3]).to(device)).sum(1, keepdim=True)], 1)
  TP['i_mouth']+= (ground[:,0,:,:] * pred[:,0,:,:]).sum().tolist()
  FP['i_mouth']+= (ground[:,1,:,:] * pred[:,0,:,:]).sum().tolist()
  TN['i_mouth']+= (ground[:,1,:,:] * pred[:,1,:,:]).sum().tolist()
  FN['i_mouth']+= (ground[:,0,:,:] * pred[:,1,:,:]).sum().tolist()

  ground = torch.cat( [batches['mouth']['labels'].index_select(1, torch.tensor([2]).to(device)), batches['mouth']['labels'].index_select(1, torch.tensor([0,1,3]).to(device)).sum(1, keepdim=True)], 1)
  pred = torch.cat( [pred_labels['mouth'].index_select(1, torch.tensor([2]).to(device)), pred_labels['mouth'].index_select(1, torch.tensor([0,1,3]).to(device)).sum(1, keepdim=True)], 1)
  TP['l_lip']+= (ground[:,0,:,:] * pred[:,0,:,:]).sum().tolist()
  FP['l_lip']+= (ground[:,1,:,:] * pred[:,0,:,:]).sum().tolist()
  TN['l_lip']+= (ground[:,1,:,:] * pred[:,1,:,:]).sum().tolist()
  FN['l_lip']+= (ground[:,0,:,:] * pred[:,1,:,:]).sum().tolist()


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

  avg_p = (PRECISION['eyebrow']+PRECISION['eye']+PRECISION['nose']+mouth_p)/4.0
  avg_r = (RECALL['eyebrow']+RECALL['eye']+RECALL['nose']+mouth_r)/4.0
  overall_F1 = 2.* avg_p*avg_r/ (avg_p+avg_r)


  print("\n\n", "PART\t\t", "F1-MEASURE ", "PRECISION ", "RECALL")
  for k in F1:
    print("%s\t\t"%k, "%.4f\t"%F1[k], "%.4f\t"%PRECISION[k], "%.4f\t"%RECALL[k])

  print("mouth(all)\t", "%.4f\t"%mouth_F1, "%.4f\t"%mouth_p, "%.4f\t"%mouth_r)
  print("Overall\t\t", "%.4f\t"%overall_F1, "%.4f\t"%avg_p, "%.4f\t"%avg_r)



def test(model, loader, criterion):
	epoch_loss = 0
	model.eval()

	with torch.no_grad():
		for batch in loader:
				
			image, labels, rects = batch['image'].to(device), batch['labels'].to(device), batch['rects'].to(device)
			rects_pred, segm, full = model(image)

			pred_labels = combine_results(rects_pred, segm, full)
			calculate_F1(pred_labels, labels)
			save_results(image, pred_labels, labels)



if __name__ == '__main__':
	test()
	show_F1()
