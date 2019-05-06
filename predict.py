import torch
import utilities
import sys
sys.path.append("./obj-dec/PyTorch-YOLOv3/")
import oracle_crop

def predict(model, dataset, device):

	model = model.to(device)

	data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)

	model.eval()
	with torch.no_grad():
		for data, target, _ in data_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
			correct = pred.eq(target.data.view_as(pred)).cpu().sum()

			print("Accuracy : %d/%d (%.6f)" % (correct, len(data_loader.dataset), 100.0*float(correct)/float(len(data_loader.dataset))))

def predict_obj(dataset, device):
	
	data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
	oracle_crop.oracle(data_loader)
