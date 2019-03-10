import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

def get_MNIST_Dataset():

	train_dataset = torchvision.datasets.MNIST('./dataset/mnist', train=True, download=True,
											   transform=torchvision.transforms.Compose([
												   torchvision.transforms.ToTensor()
											   ]))

	eval_dataset = torchvision.datasets.MNIST('./dataset/mnist', train=False, download=True,
											   transform=torchvision.transforms.Compose([
												   torchvision.transforms.ToTensor()
											   ]))

	return {'train':train_dataset,'eval':eval_dataset}

class Adv_Dataset(Dataset):
	
	def __init__(self):
		self.root_dir = "adv_samples/"
		self.all_files = os.listdir(self.root_dir)
		self.all_files.sort()

		self.size = len(self.all_files)
		self.all_labels = [int(label[-5]) for label in self.all_files]

	def __len__(self):
		return self.size

	def __getitem__(self, idx):
		img = Image.open(self.root_dir+self.all_files[idx])
		trans = torchvision.transforms.ToTensor()

		sample = (trans(img), torch.tensor(self.all_labels[idx]))
		return sample
			

def get_Adv_Dataset():
	return Adv_Dataset()