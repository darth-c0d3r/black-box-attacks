import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from utilities import *

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

		img = read_tiff_image(self.root_dir+self.all_files[idx])
		sample = (img, torch.tensor(self.all_labels[idx]))
		return sample
			

class Crop_Dataset(Dataset):

	def __init__(self, train):
		self.root_dir = "obj-dec/PyTorch-YOLOv3/crop_dataset/"
		self.train = train
		self.all_files = os.listdir(self.root_dir)
		self.all_files.sort()

		ratio = .75
		train_len = int(len(self.all_files)*ratio)

		self.train_files = self.all_files[0:train_len]
		self.eval_files = self.all_files[train_len:]

		self.train_labels = [int(file_name[file_name.index('_')+1:-5]) for file_name in self.train_files]
		self.eval_labels = [int(file_name[file_name.index('_')+1:-5]) for file_name in self.eval_files]

	def __len__(self):
		if self.train == 1:
			return len(self.train_files)
		return len(self.eval_files)

	def __getitem__(self, idx):
		img = None
		l = None
		filename = None
		if self.train == 1:
			img = Image.open(self.root_dir+self.train_files[idx])
			l = self.train_labels[idx]
			filename = self.train_files[idx]
		else:
			img = Image.open(self.root_dir+self.eval_files[idx])
			l = self.eval_labels[idx]
			filename = self.eval_files[idx]


		img = img.convert("RGB")
		img = img.resize((416,416))
		trans = torchvision.transforms.ToTensor()
		img = trans(img)
		sample = (img, torch.tensor(l),filename)
		return sample


def get_Adv_Dataset():
	return Adv_Dataset()


def get_Crop_Dataset():
	return {"train":Crop_Dataset(1), "eval":Crop_Dataset(0)}