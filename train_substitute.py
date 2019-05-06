import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import utilities
from oracle import oracle, oracle_obj
from predict import predict
from model import Classifier
import torch.utils.data as TorchUtils

def train(model, epochs, dataset, test_dataset, criterion, optimizer, device):
	
	BATCH_SIZE = 100

	for epoch in range(1, epochs+1):
		train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

		# Update (Train)
		model.train()

		for batch_idx, (data, target, _) in enumerate(train_loader):
			data, target = Variable(data.to(device)), Variable((target).to(device))

			optimizer.zero_grad()
			model.zero_grad()

			output = model(data)
			loss = criterion(output, target)

			pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
			correct = pred.eq(target.data.view_as(pred)).cpu().sum()

			loss.backward()
			optimizer.step()

		predict(model, test_dataset, device)


def create_dataset(dataset, labels): 

	data = torch.stack([img[0] for img in dataset])
	target = torch.stack([label[0] for label in labels])
	names = torch.stack([label[0] for label in labels])

	new_dataset = TorchUtils.TensorDataset(data,target,names)
	# print(len(new_dataset))
	return new_dataset

	
def augment_dataset(model, dataset, LAMBDA, device):

	new_dataset = list()
	data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

	for data, target, _ in data_loader:

		data, target = Variable(data.to(device), requires_grad=True), Variable(target.to(device))
		model.zero_grad()

		output = model(data)
		output[0][target].backward()

		data_new = data[0] + LAMBDA * torch.sign(data.grad.data[0])
		new_dataset.append(data_new.cpu())

	new_dataset = torch.stack([data_point for data_point in new_dataset])
	new_dataset = TorchUtils.TensorDataset(new_dataset)

	new_dataset = TorchUtils.ConcatDataset([dataset, new_dataset])
	return new_dataset



def train_substitute(oracle_model, dataset, test_dataset, device, MAX_RHO, LAMBDA, EPOCHS):

	if oracle_model is not None: oracle_model = oracle_model.to(device)

	n_classes = 81
	input_shape = list(dataset[0][0].shape)

	conv = [input_shape[0],4,8,16,32]
	fc = [512,256,128]

	model = None
	for rho in range(MAX_RHO):

		input_shape = list(dataset[0][0].shape)

		dummy_labels = None

		if oracle_model is not None:
			dummy_labels = oracle(oracle_model, dataset, device)
		else:
			dummy_labels = oracle_obj(dataset, device)

		dummy_dataset = create_dataset(dataset, dummy_labels)

		model = Classifier(input_shape, conv, fc, n_classes).to(device)
		criterion = nn.CrossEntropyLoss().to(device)
		optimizer = optim.Adagrad(model.parameters(), lr=0.01)

		train(model, EPOCHS, dummy_dataset, test_dataset, criterion, optimizer, device)
		print("Rho: %d"%(rho))
		print("Dataset Size: %d"%(len(dataset)))

		dataset = augment_dataset(model, dummy_dataset, LAMBDA, device)
	
	return model

def train_substitute_not_scratch(oracle_model, dataset, test_dataset, device, MAX_RHO, LAMBDA, EPOCHS): 

	if oracle_model is not None: oracle_model = oracle_model.to(device)

	n_classes = 81
	input_shape = list(dataset[0][0].shape)

	conv = [input_shape[0], 4, 8, 16, 32]
	fc = [512,156,128]

	model = Classifier(input_shape, conv, fc, n_classes).to(device)
	for rho in range(MAX_RHO):

		input_shape = list(dataset[0][0].shape)

		dummy_labels = None

		if oracle_model is not None:
			dummy_labels = oracle(oracle_model, dataset, device)
		else:
			dummy_labels = oracle_obj(dataset, device)

		dummy_dataset = create_dataset(dataset, dummy_labels)

		criterion = nn.CrossEntropyLoss().to(device)
		optimizer = optim.Adagrad(model.parameters(), lr=0.01)

		train(model, EPOCHS, dummy_dataset, test_dataset, criterion, optimizer, device)
		print("Rho: %d"%(rho))
		print("Dataset Size: %d"%(len(dataset)))

		dataset = augment_dataset(model, dummy_dataset, LAMBDA, device)

	return model
