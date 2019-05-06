import torch
import utilities
import sys
sys.path.append("./obj-dec/PyTorch-YOLOv3/")
import oracle_crop

def oracle(model, data, device):

	model = model.to(device)

	data_loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False)
	pred = None

	model.eval()
	with torch.no_grad():
		for data in data_loader:
				data = data[0].to(device)
				output = model(data)
				pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
	
	return pred

def oracle_obj(data, device):

	data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
	pred = oracle_crop.oracle(data_loader)
	return pred