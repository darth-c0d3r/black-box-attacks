import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import utils

def adv_sample(model_name, data, epsilon, target):
	'''
	data contains (image, target_original_label) from a dataset class
	target is the class to which the data is being misclassified
	'''
	# BATCH_SIZE = 1
	EPOCHS = 6
	samples = torch.zeros(data.shape)

	L2_loss = nn.MSELoss().to(device)
	Classification_loss = nn.CrossEntropyLoss().to(device)

	device = utils.get_device(1)
	model = torch.load("saved_models/"+model_name).to(device)
	data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)

	idx = 0
	for data, label in data_loader:
		data = data.to(device)
		sample, target = Variable(data.data, requires_grad=True), Variable(torch.tensor(target)).to(device)

		for epoch in range(EPOCHS):
			loss = Classification_loss(sample, target) + lambda_img * L2_loss(sample, data)
			model.zero_grad()
			loss.backward()

			sample = sample - epsilon * torch.sign(sample.grad.sample)
			sample = (sample - torch.min(sample)) / (torch.max(sample) - torch.min(sample))

		samples[idx] = sample[0]

	return samples