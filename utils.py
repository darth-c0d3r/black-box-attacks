from PIL import Image
import torchvision
import torch
import os

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