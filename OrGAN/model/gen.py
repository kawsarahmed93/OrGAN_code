""" Full assembly of the parts to form the complete network """

from .gen_parts import *
import torch.nn as nn



class OrGAN(nn.Module):
	def __init__(self, bilinear=False):
		super(OrGAN, self).__init__()
		self.bilinear = bilinear

		self.inc = (DoubleConv(1, 64))
		self.down1 = (Down(64, 128))
		self.down2 = (Down(128, 256))
		self.down3 = (Down(256, 512))
		factor = 2 if bilinear else 1
		self.down4 = (Down(512, 1024 // factor))
		self.up1 = (Up(1024, 512 // factor, bilinear))
		self.up2 = (Up(512, 256 // factor, bilinear))
		self.up3 = (Up(256, 128 // factor, bilinear))
		self.up4 = (Up(128, 64, bilinear))
		self.up5 = (DoubleConv(64, 4, 32))
		self.outc = (OutConv(4, 1))

		self.domain_classifier = (DomainClassifier(2))

	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		x = self.up5(x)
		c = x
		out = self.outc(x)
		c = c.view(c.size(0), -1)
		logit = self.domain_classifier(c)
		return out, logit

	def use_checkpointing(self):
		self.inc = torch.utils.checkpoint(self.inc)
		self.down1 = torch.utils.checkpoint(self.down1)
		self.down2 = torch.utils.checkpoint(self.down2)
		self.down3 = torch.utils.checkpoint(self.down3)
		self.down4 = torch.utils.checkpoint(self.down4)
		self.up1 = torch.utils.checkpoint(self.up1)
		self.up2 = torch.utils.checkpoint(self.up2)
		self.up3 = torch.utils.checkpoint(self.up3)
		self.up4 = torch.utils.checkpoint(self.up4)
		self.outc = torch.utils.checkpoint(self.outc)
		self.domain_classifier = torch.utils.checkpoint(self.domain_classifier)