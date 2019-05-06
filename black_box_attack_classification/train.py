import torch
from model import Classifier
import utils
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from predict import predict

# hyper-parameters

def train(model, dataset, criterion, optimizer, device, EPOCHS, BATCH_SIZE):

	REPORT_EVERY = (len(dataset["train"]) // BATCH_SIZE) // 5

	for epoch in range(1,EPOCHS+1):
		train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=BATCH_SIZE, shuffle=True)

		# Update (Train)
		model.train()

		for batch_idx, (data, target) in enumerate(train_loader):
			data, target = Variable(data.to(device)), Variable((target).to(device))
			optimizer.zero_grad()
			model.zero_grad()
			output = model(data)
			loss = criterion(output,target)
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
		predict(model, dataset['eval'], device)
