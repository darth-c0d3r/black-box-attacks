import torch
import torch.nn as nn

class Classifier(nn.Module):

	def __init__(self, size, conv, fc, n_classes):

		super(Classifier, self).__init__()

		self.dropout_rate = 0.5
		self.n_classes = n_classes
		self.input_size = size[:]

		# convolutional layers
		kernel_size = [5] * (len(conv)-1)
		stride = [1] * (len(conv)-1)
		# stride[0]=2
		padding = [1] * (len(conv)-1)

		self.conv_layers = nn.ModuleList()

		for i in range(len(conv)-1):
			self.conv_layers.append(nn.Conv2d(conv[i], conv[i+1], kernel_size[i], stride[i], padding[i]))
			size[0] = conv[i+1]
			size[1] = (size[1] + 2*padding[i] - kernel_size[i])//stride[i] + 1
			size[2] = (size[2] + 2*padding[i] - kernel_size[i])//stride[i] + 1

		self.size = size[0] * size[1] * size[2]
		print("Input Size to FC : %d" % (self.size))

		# fully connected layers
		fc = [self.size] + fc
		self.fc_layers = nn.ModuleList()
		self.dropout_layers = nn.ModuleList()

		for i in range(len(fc)-1):
			self.fc_layers.append(nn.Linear(fc[i], fc[i+1]))
			self.dropout_layers.append(nn.Dropout(self.dropout_rate))

		# output layer
		self.output_layer = nn.Linear(fc[-1], n_classes)

	def forward(self, x):
		index=0

		# convolutional layers
		for conv_layer in self.conv_layers:
			x = torch.relu(conv_layer(x))

		# fully connected layers
		x = x.view(-1, self.size)
		for fc_layer in self.fc_layers:
			x = torch.relu(fc_layer(x))
			x = self.dropout_layers[index](x)
			index += 1

		# output layer
		x = self.output_layer(x)

		return x
