import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import utils
from torch.nn import Parameter

import misc


class MLP(nn.Module):
	"""
	A small multilayer perceptron with parameters that we can optimize for the task.
	"""
	def __init__(self, num_features=784, num_hidden=64, num_outputs=10):
		super(MLP, self).__init__()

		self.W_1 = Parameter(init.xavier_normal_(torch.Tensor(num_hidden, num_features)))
		self.b_1 = Parameter(init.constant_(torch.Tensor(num_hidden), 0))

		self.W_2 = Parameter(init.xavier_normal_(torch.Tensor(num_outputs, num_hidden)))
		self.b_2 = Parameter(init.constant_(torch.Tensor(num_outputs), 0))

	def forward(self, x):
		x = F.relu(F.linear(x, self.W_1, self.b_1))
		x = F.linear(x, self.W_2, self.b_2)

		return x


class CNN(nn.Module):
	"""
	A small convolutional neural network with parameters that we can optimize for the task.
	"""
	def __init__(self, num_layers=4, num_filters=64, num_classes=10, input_size=(3, 32, 32)):
		super(CNN, self).__init__()

		self.channels = input_size[0]
		self.height = input_size[1]
		self.width = input_size[2]
		self.num_filters = num_filters

		self.conv_in = nn.Conv2d(self.channels, self.num_filters, kernel_size=5, padding=2)
		cnn = []
		for _ in range(num_layers):
			cnn.append(nn.Conv2d(self.num_filters, self.num_filters, kernel_size=3, padding=1))
			cnn.append(nn.BatchNorm2d(self.num_filters))
			cnn.append(nn.ReLU())
		self.cnn = nn.Sequential(*cnn)

		self.out_lin = nn.Linear(self.num_filters*self.width*self.height, num_classes)

		if torch.cuda.is_available():
			self.cuda()


	def forward(self, x):
		if torch.cuda.is_available():
			x = x.cuda()

		x = F.relu(self.conv_in(x))
		x = self.cnn(x)
		x = x.reshape(x.size(0), -1)

		return self.out_lin(x)


def fit(net, data, optimizer, batch_size=128, num_epochs=250, lr_schedule=False):
	"""
	Fits parameters of a network `net` using `data` as training data and a given `optimizer`.
	"""
	train_generator = utils.data.DataLoader(data[0], batch_size=batch_size)
	val_generator = utils.data.DataLoader(data[1], batch_size=batch_size)

	losses = misc.AvgLoss()
	val_losses = misc.AvgLoss()

	if lr_schedule:
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

	for epoch in range(num_epochs+1):

		epoch_loss = misc.AvgLoss()
		epoch_val_loss = misc.AvgLoss()
		
		# For tracking accuracy
		val_correct = 0
		val_total = 0
		
		# Calculate validation loss and accuracy
		with torch.no_grad():
			for x, y in val_generator:
				y = y.type(torch.LongTensor)
				if torch.cuda.is_available(): y = y.cuda()
				outputs = net(x)
				epoch_val_loss += F.cross_entropy(outputs, y).cpu()
				
				# Calculate accuracy
				_, predicted = torch.max(outputs.data, 1)
				val_total += y.size(0)
				val_correct += (predicted == y).sum().item()

		# For tracking training accuracy
		train_correct = 0
		train_total = 0
		
		for x, y in train_generator:
			y = y.type(torch.LongTensor)
			if torch.cuda.is_available(): y = y.cuda()
			
			# Forward pass
			outputs = net(x)
			loss = F.cross_entropy(outputs, y).cpu()
			epoch_loss += loss
			
			# Calculate accuracy
			_, predicted = torch.max(outputs.data, 1)
			train_total += y.size(0)
			train_correct += (predicted == y).sum().item()
			
			# Backward pass and optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		if lr_schedule:
			scheduler.step(epoch_loss.avg)

		# Calculate accuracies
		train_accuracy = 100 * train_correct / train_total
		val_accuracy = 100 * val_correct / val_total
		
		# Print progress for every epoch
		print(f'Epoch {epoch}/{num_epochs}, loss: {epoch_loss}, val loss: {epoch_val_loss}, '
			  f'train acc: {train_accuracy:.2f}%, val acc: {val_accuracy:.2f}%')

		losses += epoch_loss.losses
		val_losses += epoch_val_loss.losses

	return losses, val_losses
