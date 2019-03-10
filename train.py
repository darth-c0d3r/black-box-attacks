import torch
from model import Classifier
from dataset import get_MNIST_Dataset
import utils
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from white_box import *
from predict import predict

# hyper-parameters

EPOCHS = 20
BATCH_SIZE = 1000
REPORT_EVERY = 12

def train(model, dataset, criterion, optimizer, device):

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
		predict(model, dataset['eval'])
		# model.eval()
		# eval_loss = float(0)
		# correct = 0
		# eval_loader = torch.utils.data.DataLoader(dataset['eval'], batch_size=BATCH_SIZE, shuffle=False)

		# with torch.no_grad():
		# 	for data, target in eval_loader:
		# 		data, target = Variable(data.to(device)), Variable((target).to(device))
		# 		output = model(data)
		# 		eval_loss += len(data) * criterion(output, target).item() # sum up batch loss
		# 		pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
		# 		correct += pred.eq(target.data.view_as(pred)).cpu().sum()

		# eval_loss /= len(eval_loader.dataset)
		# accuracy = float(correct) / len(eval_loader.dataset)

		# print('Eval Epoch: {} [{}/{} ({:.6f}%)] Loss: {:.6f}, Accuracy: {}/{} ({:.6f})'.format(
		# 			epoch, len(eval_loader.dataset), len(eval_loader.dataset),
		# 			100.0, eval_loss, correct, len(eval_loader.dataset), accuracy))




def main():

	device = utils.get_device(1)

	dataset = get_MNIST_Dataset()
	utils.print_dataset_details(dataset)

	input_shape = list(dataset["train"][0][0].shape)

	conv = [1, 4, 8, 16, 32]
	fc = []
	n_classes = 10

	model = Classifier(input_shape, conv, fc, n_classes).to(device)
	criterion = nn.CrossEntropyLoss().to(device)

	# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	optimizer = optim.Adagrad(model.parameters(), lr=0.01)

	train(model, dataset, criterion, optimizer, device)
	utils.save_model(model)

if __name__ == '__main__':
	main()