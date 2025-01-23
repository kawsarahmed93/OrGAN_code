import torch 
import torch.nn as nn 


class CNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = nn.Sequential(
				nn.Conv2d(1, 16, kernel_size=(1,1), stride = (1,1), padding = (0,0), bias=False),
				nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(16, 32, kernel_size=(1,1), stride = (1,1), padding = (0,0), bias=False),
				nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(32, 64, kernel_size=(1,1), stride = (1,1), padding = (0,0), bias=False),
				nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(64, 32, kernel_size=(1,1), stride = (1,1), padding = (0,0), bias=False),
				nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(32, 16, kernel_size=(1,1), stride = (1,1), padding = (0,0), bias=False),
				nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(16, 1, kernel_size=(1,1), stride = (1,1), padding = (0,0), bias=False),
				nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
				# nn.LeakyReLU(0.2,inplace=True)
				nn.Sigmoid()
		)

	def forward(self, data):
		x = self.model(data)
		return x