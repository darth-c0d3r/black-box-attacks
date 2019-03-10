import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import utils
from oracle import oracle
from predict import predict

def train(model, epochs, dataset, criterion, optimizer, device): # dataset has images and their labels
	
	REPORT_EVERY = len(dataset) // 3
	BATCH_SIZE = 100

	for epoch in range(1, epochs+1):
		train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

		# Update (Train)
		model.train()

		for batch_idx, (data, target) in enumerate(train_loader):
			data, target = Variable(data.to(device)), Variable((target).to(device))

			optimizer.zero_grad()
			model.zero_grad()

			output = model(data)
			loss = criterion(output, target)

			pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
			correct = pred.eq(target.data.view_as(pred)).cpu().sum()

			loss.backward()
			optimizer.step()

			if batch_idx % REPORT_EVERY == REPORT_EVERY-1:
				print('Train Epoch: {} [{}/{} ({:.6f}%)] Loss: {:.6f}, Accuracy: {}/{} ({:.6f})'.format(
					epoch, (batch_idx * BATCH_SIZE) + len(data), len(train_loader.dataset),
					float((batch_idx * BATCH_SIZE) + len(data))*100.0 / float(len(train_loader.dataset)), 
					loss.item(), correct, len(data), float(correct)/float(len(data))))

		# Evaluate
		predict(model, dataset)


def create_dataset(dataset, labels): # dataset has no target labels

	

def train_substitute(oracle, dataset): # dataset has no target labels

	device = utils.get_device(1)

	MAX_RHO = 5
	LAMBDA = 0.3
	EPOCHS = 10

	input_shape = list(init_dataset[0][0].shape)
	n_classes = 10

	conv = [input_shape[0], 4, 8, 16, 32]
	fc = []

	model = None
	for rho in MAX_RHO:

		dummy_labels = oracle(oracle_name, dataset)
		dummy_dataset = create_dataset(dataset, dummy_labels)

		model = Classifier(input_shape, conv, fc, n_classes).to(device)
		criterion = nn.CrossEntropyLoss().to(device)
		optimizer = optim.Adagrad(model.parameters(), lr=0.01)

		train(model, EPOCHS, dummy_dataset, criterion, optimizer, device)

		dataset = augment_dataset(dummy_dataset, LAMBDA)

	return model