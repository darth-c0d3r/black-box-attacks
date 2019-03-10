import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import utils

def adv_sample(model_name, dataset, target):
	'''
	data contains (image, target_original_label) from a dataset class
	target is the class to which the data is being misclassified
	'''
	device = utils.get_device(1)

	EPOCHS = 1000
	LAMBDA = 20.0
	EPSILON = 0.5
	NUM_SAMPLES = 10

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
		sample, target = Variable(torch.randn(([1]+model.input_size)).to(device), requires_grad=True), Variable(target).to(device)

		# sample = (sample - torch.min(sample)) / (torch.max(sample) - torch.min(sample))
		sample = torch.clamp(sample,0,1)

		for epoch in range(EPOCHS):

			sample, target = Variable(sample.data, requires_grad=True).to(device), Variable(target).to(device)
			output = torch.sigmoid(model(sample))
			loss = L2_loss(output, target) + LAMBDA * L2_loss(sample, data)
			model.zero_grad()
			loss.backward()

			sample = sample - EPSILON * sample.grad.data

			# sample = (sample - torch.min(sample)) / (torch.max(sample) - torch.min(sample))
			sample = torch.clamp(sample,0,1)

		samples[idx] = sample[0]
		idx += 1

		if idx == NUM_SAMPLES:
			break

	for i in range(5):
		inp = int(input("idx: "))
		utils.disp_img(samples[inp], (100,100))
		utils.disp_img(dataset[inp][0], (100,100))

	return samples