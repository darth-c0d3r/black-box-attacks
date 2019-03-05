import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import utils

def adv_sample(model_name, dataset, epsilon, target):
	'''
	data contains (image, target_original_label) from a dataset class
	target is the class to which the data is being misclassified
	'''
	# BATCH_SIZE = 1
	device = utils.get_device(1)

	EPOCHS = 5
	lambda_img = 20.0
	samples = []

	L2_loss = nn.MSELoss().to(device)
	Classification_loss = nn.CrossEntropyLoss().to(device)
	
	model = torch.load("saved_models/"+model_name).to(device)
	data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

	idx = 0
	for data, label in data_loader:

		data = data.to(device)
		sample, target = Variable(data.data, requires_grad=True), Variable(torch.tensor([target])).to(device)
		sample = (sample - torch.min(sample)) / (torch.max(sample) - torch.min(sample))

		for epoch in range(EPOCHS):

			sample = Variable(sample.data, requires_grad=True)
			output = model(sample)
			loss = Classification_loss(output, target) + lambda_img * L2_loss(sample, data)
			model.zero_grad()
			loss.backward()

			sample = sample - epsilon * sample.grad.data
			sample = (sample - torch.min(sample)) / (torch.max(sample) - torch.min(sample))

		samples.append(sample[0])
		idx += 1

		print(model(sample))


		break


	# print(type(samples[0]))

	utils.disp_img(samples[0], (100,100))
	utils.disp_img(dataset[0][0], (100,100))

	return samples