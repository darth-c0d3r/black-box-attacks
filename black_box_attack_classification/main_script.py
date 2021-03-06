from dataset import *
from model import *
import train as tr
import train_substitute as trs
import utils
from white_box import *
from predict import predict
from oracle import oracle

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as TorchUtils
import argparse
import random

# USAGE_STRING = """Arguments:\n(a) -i /path/to/input.bin\n(b) -t /path/to/target.bin\n(c) -ig /path/to/gradInput.bin""" 

parser = argparse.ArgumentParser()

parser.add_argument("--bb", help="Train Black box Initially", dest='bb',default=False,action='store_true')
parser.add_argument("--sub", help="Train Substitute Model", dest='sub',default=False,action='store_true')
parser.add_argument("--adv", help="Generate Adversarial Samples", dest='adv',default=False,action='store_true')
parser.add_argument("--test", help="Test the model on Adversarial Samples", dest='test',default=False,action='store_true')

_a = parser.parse_args()
args = {}

for a in vars(_a):
	args[a] = getattr(_a, a)

device = utils.get_device(1)

dataset = get_MNIST_Dataset()
utils.print_dataset_details(dataset)

if args['bb']:
	input_shape = list(dataset["train"][0][0].shape)

	conv = [input_shape[0]]
	fc = []
	n_classes = 10

	epochs = 20 # int(input("Epochs: "))
	batch_size = 1000 # int(input("batch_size: "))
	lr = 0.01 # float(input("lr: "))

	model = Classifier(input_shape, conv, fc, n_classes).to(device)
	criterion = nn.CrossEntropyLoss().to(device)

	# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	optimizer = optim.Adagrad(model.parameters(), lr=lr)

	tr.train(model, dataset, criterion, optimizer, device, epochs, batch_size)

	print("Saving Black Box:")
	utils.save_model(model)


if args['sub']:
	bb = input("Black box model name: ")

	oracle_model = torch.load("saved_models/"+bb)

	test_dataset = dataset["eval"]
	dataset = dataset["train"]

	num_points = 150 # int(input("Num Points Initial: "))
	idxs = []
	for _ in range(num_points):
		while True:
			idx = random.randint(0,len(dataset)-1)
			if idx not in idxs:
				idxs.append(idx)
				break

	dataset = TorchUtils.Subset(dataset, idxs)

	max_rho = 6 # int(input("Max_Rho: "))
	lamba = 0.1 # float(input("Lambda: "))
	epochs = 10 # int(input("Epochs: "))

	scratch = input("From scratch or not [y/n]: ")
	model = None
	if(scratch == 'y'): model = trs.train_substitute(oracle_model, dataset, test_dataset, device, max_rho, lamba, epochs)
	else: model = trs.train_substitute_not_scratch(oracle_model, dataset, test_dataset, device, max_rho, lamba, epochs)

	print("Saving Substitute Model:")
	utils.save_model(model)

if args['adv']:
	model_name = input("Substitute Model Name: ")
	target = 6 # int(input("Directed Label to misclassify: "))
	num_samples = 20

	samples = adv_sample(model_name, dataset['eval'], target, num_samples)

	utils.save_tiff_images(samples, target)


if args['test']:
	model_name = input("Test Model Name: ")
	model = torch.load("saved_models/"+model_name)
	data = get_Adv_Dataset()
	print(data[0][0].shape)
	
	print("For", model_name[:-3], "model: ")
	predict(model, data, device)