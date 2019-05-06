import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import utilities
import numpy as np

def adv_sample(model_name, dataset, target, num_samples):
	'''
	data contains (image, target_original_label) from a dataset class
	target is the class to which the data is being misclassified
	'''
	device = utilities.get_device(1)

	EPOCHS = 5000
	LAMBDA = 10.0
	EPSILON = 0.25
	NUM_SAMPLES = num_samples

	L2_loss = nn.MSELoss().to(device)
	Classification_loss = nn.CrossEntropyLoss().to(device)
	
	model = None
	if device == utilities.get_device(0):
		model = torch.load("saved_models/"+model_name, map_location='cpu').to(device)
	else:
		model = torch.load("saved_models/"+model_name).to(device)
	data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

	idx = 0
	samples = torch.zeros([NUM_SAMPLES]+list(dataset[0][0].shape))

	# init target
	target_ = target
	target = torch.zeros((1,model.n_classes))
	target[0][target_] = 1
	names = []
	for data, label, name in data_loader:

		data = data.to(device)
		sample, target = Variable(data.to(device), requires_grad=True), Variable(target).to(device)

		sample = (sample - torch.min(sample)) / (torch.max(sample) - torch.min(sample))
		# sample = torch.clamp(sample,0,1)

		for epoch in range(EPOCHS):

			sample, target = Variable(sample.data, requires_grad=True).to(device), Variable(target).to(device)
			output = torch.sigmoid(model(sample))
			loss = L2_loss(output, target) + LAMBDA * L2_loss(sample, data)
			model.zero_grad()
			loss.backward()

			sample = sample - EPSILON * sample.grad.data

			sample = (sample - torch.min(sample)) / (torch.max(sample) - torch.min(sample))
			# sample = torch.clamp(sample,0,1)

		names.append(name[0])
		samples[idx] = sample[0]
		output = model(sample)
		pred = output.data.max(1, keepdim=True)[1]
		print(pred[0][0].item(), label.item()) # predicted by our model, predicted by oracle
		idx += 1

		if idx == NUM_SAMPLES:
			break

	return samples, names

def adv_sample_papernot(model_name, dataset, target):
	'''
	data contains (image, target_original_label) from a dataset class
	target is the class to which the data is being misclassified
	'''
	device = utilities.get_device(1)

	EPOCHS = 10
	LAMBDA = 20.0
	NUM_SAMPLES = 10
	EPSILON = 0.5

	samples = []

	L2_loss = nn.MSELoss().to(device)
	Classification_loss = nn.CrossEntropyLoss().to(device)
	
	model = torch.load("saved_models/"+model_name).to(device)
	data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

	idx = 0
	samples = torch.zeros([NUM_SAMPLES]+list(dataset[0][0].shape))

	# init target
	target_ = target
	target = torch.zeros((1,model.n_classes))
	target[0][target_] = 1

	for data, label in data_loader:

		data = data.to(device)
		sample, target = Variable(data.data.to(device), requires_grad=True), Variable(target).to(device)

		sample = torch.clamp(sample,0,1)

		for epoch in range(EPOCHS):

			sample, target = Variable(sample.data, requires_grad=True).to(device), Variable(target).to(device)
			output = torch.sigmoid(model(sample))
			loss = L2_loss(output, target) + LAMBDA * L2_loss(sample, data)
			model.zero_grad()
			loss.backward()

			delta = EPSILON * torch.sign(sample.grad.data)

			sample = Variable(sample.data, requires_grad=True).to(device)
			output = model(sample)
			model.zero_grad()
			output[0][target_].backward(retain_graph=True)

			jacobian_t = sample.grad.data
			jacobian_non_t = torch.zeros(jacobian_t.shape)
			for i in range(model.n_classes):
				if i == target_:
					continue
				model.zero_grad()
				output[0][i].backward(retain_graph=True)
				jacobian_non_t += sample.grad.data

			saliency = np.zeros(sample.reshape(-1).shape)
			for i in range(sample.shape[2]):
				for j in range(sample.shape[3]):
					if jacobian_t[0][0][i][j]<0 or jacobian_non_t[0][0][i][j]>0:
						continue
					saliency[i*sample.shape[3]+j] = jacobian_t[0][0][i][j]*abs(jacobian_non_t[0][0][i][j])

			indices = np.argsort(saliency)

			with torch.no_grad():
				for i in range(len(indices)-1,-1,-1):
					sample, target = Variable(sample.data).to(device), Variable(target).to(device)
					output = model(sample)
					pred = output.data.max(1, keepdim=True)[1]
					if torch.eq(pred[0][0],target[0][target_].long()):
						print("Done")
						break
					sample[0][0][indices[i]//sample.shape[3]][indices[i]%sample.shape[3]] -= delta[0][0][indices[i]//sample.shape[3]][indices[i]%sample.shape[3]]
					sample = (sample-torch.min(sample))/(torch.max(sample)-torch.min(sample))	

		samples[idx] = sample[0]
		idx += 1

		if idx == NUM_SAMPLES:
			break

	return samples
