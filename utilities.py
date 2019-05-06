from PIL import Image
import torchvision
import torch
import os
from libtiff import TIFF

def disp_img(img, size=None):
	trans = torchvision.transforms.ToPILImage()
	img = trans(img)
	if size is not None:
		img = img.resize(size)
	img.show()

def print_dataset_details(dataset):
	print("Training Dataset Size : %d" % (len(dataset["train"])))
	print("Eval Dataset Size : %d" % (len(dataset["eval"])))
	print("Image Shape : " + str(dataset["train"][0][0].shape))

def get_device(cuda):
	device = torch.device("cuda" if torch.cuda.is_available() and cuda == 1 else "cpu") # default gpu
	print("Device:", device)
	return device

def save_model(model):
	folder = "saved_models/"
	files = os.listdir(folder)

	while True:
		filename = input("Enter filename : ")
		if filename in files:
			response = input("Warning! File already exists. Override? (y/n) : ")
			if response.strip() in ("Y", "y"):
				break
			else:
				continue
		break

	torch.save(model, folder+filename)


def save_tiff_image(img, name, folder):
	if folder not in os.listdir():
		os.mkdir(folder)

	img = img.detach().numpy()
	print(img.shape)
	tiff = TIFF.open(folder+"/"+name, 'w')
	tiff.write_image(img,write_rgb=True)
	tiff.close()

	# trans = torchvision.transforms.ToPILImage()
	# img = trans(img)
	# img.save(folder+name)

def save_tiff_images(images, target, names):
	for i in range(images.shape[0]):
		name = names[i][:-5]
		print(name)
		save_tiff_image(images[i], "%s.tif_%d"%(name,target), "adv_samples")

def read_tiff_image(file):
	tiff = TIFF.open(file, 'r')
	img = tiff.read_image()
	print(img.shape)
	return torch.tensor(img)

def show_tiff_image(file):
	img = read_tiff_image(file)
	trans = torchvision.transforms.ToPILImage()
	print(img.shape)
	img = trans(img)
	img.show()