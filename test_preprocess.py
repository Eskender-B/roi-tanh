from preprocess import ToTensor, ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils


test_dataset = ImageDataset(txt_file='testing.txt',
                                           root_dir='data/SmithCVPR2013_dataset_resized',
                                           bg_indexs=set([0]),
                                           transform=transforms.Compose([
                                               ToTensor(),
                                           ]))
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

print("here")
for batch in test_loader:
	print(batch['rects'])
