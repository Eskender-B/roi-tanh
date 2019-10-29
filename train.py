from preprocess import Rescale, ToTensor, ImageDataset, DataArg
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
                                           root_dir='data/SmithCVPR2013_dataset_resized',
                                           bg_indexs=set([0]),
                                           transform=transforms.Compose([                                               ,
                                               ToTensor()
                                           ]))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)


valid_dataset = ImageDataset(txt_file='tuning.txt',
                                           root_dir='data/SmithCVPR2013_dataset_resized',
                                           bg_indexs=set([0,1,10]),
                                           transform=transforms.Compose([
                                               ToTensor()
                                           ]))
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)


test_dataset = ImageDataset(txt_file='testing.txt',
                                           root_dir='data/SmithCVPR2013_dataset_resized',
                                           bg_indexs=set([0,1,10]),
                                           transform=transforms.Compose([
                                               ToTensor(),
                                           ]))
test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)



model = ICNN(output_maps=9)
optimizer = optim.Adam(model.parameters(), lr=args.lr)



def train1(epoch, model, train_loader, optimizer, criterion):
	loss_list = []
	model.train()

	for i, batch in enumerate(train_loader):
		optimizer.zero_grad()
		image, labels = batch['image'].to(device), batch['labels'].to(device)
		predictions = model(image)
		loss = criterion(predictions, labels.argmax(dim=1, keepdim=False))

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
		image, labels = batch['image'].to(device), batch['labels'].to(device)
		predictions = model(image)
		loss = criterion(predictions, labels.argmax(dim=1, keepdim=False))

		loss.backward()
		optimizer.step()


		loss_list.append(loss.item())


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
			image, labels = batch['image'].to(device), batch['labels'].to(device)
			rect, segm, fcn = model(image)
			loss = criterion(predictions, labels.argmax(dim=1, keepdim=False))

			epoch_loss += loss.item()

	return epoch_loss / len(loader)


def evaluate2(model, loader, criterion):
	epoch_loss = 0
	model.eval()

	with torch.no_grad():
		for batch in loader:
			image, labels = batch['image'].to(device), batch['labels'].to(device)
			predictions = model(image)
			loss = criterion(predictions, labels.argmax(dim=1, keepdim=False))

			epoch_loss += loss.item()

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

	scheduler.step()
	msg = '...Epoch %02d, val loss = %.4f' % (epoch, valid_loss)
	LOG_INFO(msg)


model = pickle.load(open('res/saved-model.pth', 'rb'))


msg = 'Min @ Epoch %02d, val loss = %.4f' % (epoch_min, LOSS)
LOG_INFO(msg)
test_loss = evaluate2(model, test_loader, criterion)
LOG_INFO('Finally, test loss = %.4f' % (test_loss))