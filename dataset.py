import torch
import torchvision

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