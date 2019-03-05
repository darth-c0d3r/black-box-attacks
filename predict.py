import torch
import utils

def predict(model_name, data):

	device = utils.get_device(1)
	model = torch.load("saved_models/"+model_name).to(device)

	data_loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False)

	model.eval()
	with torch.no_grad():
		for data, target in data_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
			correct = pred.eq(target.data.view_as(pred)).cpu().sum()

			print("Accuracy : %d/%d (%.6f)" % (correct, len(data_loader.dataset), 100.0*float(correct)/float(len(data_loader.dataset))))