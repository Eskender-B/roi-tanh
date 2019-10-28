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
