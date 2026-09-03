import numpy as np
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
import random
import pandas as pd
import matplotlib.pyplot as plt 
import imageio
from scipy import ndimage
from skimage import exposure
import pydicom as di
import itertools
import random
import numpy as np
import torch
from torch.utils.data.sampler import Sampler

class TwoStreamBatchSampler(Sampler):
	"""Iterate two sets of indices

	An 'epoch' is one iteration through the primary indices.
	During the epoch, the secondary indices are iterated through
	as many times as needed.
	"""

	def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
		self.primary_indices = primary_indices
		self.secondary_indices = secondary_indices
		self.secondary_batch_size = secondary_batch_size
		self.primary_batch_size = batch_size - secondary_batch_size

		assert len(self.primary_indices) >= self.primary_batch_size > 0
		assert len(self.secondary_indices) >= self.secondary_batch_size > 0

	def __iter__(self):
		primary_iter = iterate_once(self.primary_indices)
		secondary_iter = iterate_eternally(self.secondary_indices)
		return (
			primary_batch + secondary_batch
			for (primary_batch, secondary_batch)
			in zip(grouper(primary_iter, self.primary_batch_size),
				   grouper(secondary_iter, self.secondary_batch_size))
		)

	def __len__(self):
		return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
	return np.random.permutation(iterable)


def iterate_eternally(indices):
	def infinite_shuffles():
		while True:
			yield np.random.permutation(indices)

	return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
	"Collect data into fixed-length chunks or blocks"
	# grouper('ABCDEFG', 3) --> ABC DEF"
	args = [iter(iterable)] * n
	return zip(*args)

class XrayDataset(Dataset):
	def __init__(self, images_filenames, images_directory, l_id, transform=None, r_transform=None, MS=False):
		self.images_filenames = images_filenames
		self.images_directory = images_directory
		self.transform = transform
		self.r_transform = r_transform
		self.count = 0
		self.l_id = l_id
		self.mean_std = [1.015, 0.345, 0.574, 0.257] #  Luna16 train, Vin-Dr train
		self.MS=MS


	def __len__(self):
		return len(self.images_filenames)

	def __getitem__(self, idx):

		image_filename = self.images_filenames[idx]
		lungs_name  = image_filename.replace('Xray','Lungs')
		s_flag = "Luna16" in image_filename

		if idx < self.l_id: 
			check = True
		else:
			check = False


		image = np.load(os.path.join(self.images_directory, image_filename))
		if check:

			lungs = np.load(os.path.join(self.images_directory.replace('Xray','Lungs'), lungs_name))

			if s_flag:

				# m = image.max()

				# image = image
				# lungs = lungs

				image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

				if self.transform is not None:
					transformed = self.transform(image=image, mask=lungs)
					image = transformed["image"]
					lungs = transformed["mask"]

				image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

				if self.MS:
					image, mc,sc = standardize(image,self.mean_std[0],self.mean_std[1])
				else:
					image, mc,sc = standardize(image,0,0)
					
				lungs = normal(lungs)
				masks = lungs

			else:
				masks = lungs
			
		else:
			# mn = image.min()
			
			image = np.float32(image)
			masks = image

			image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

			if self.r_transform is not None:
				transformed = self.r_transform(image=image, mask=masks)
				image = transformed["image"]
				masks = transformed["mask"]
			image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

			if self.MS:
				image, mc,sc = standardize(image,self.mean_std[2],self.mean_std[3])
			else:
				image, mc,sc = standardize(image,0,0)

		image = np.expand_dims(image, axis=2)
		image = np.transpose(image, (2, 0, 1))

		return image, masks

class TXDataset(Dataset):
	def __init__(self, images_filenames, images_directory, transform=None, MS=False):
		self.images_filenames = images_filenames
		self.images_directory = images_directory
		self.transform = transform
		self.count = 0
		self.mean_std = [1.015, 0.345] #  Luna16 train
		self.MS=MS

	def __len__(self):
		return len(self.images_filenames)

	def __getitem__(self, idx):

		image_filename = self.images_filenames[idx]
		lungs_name  = image_filename.replace('Xray','Lungs')
		s_flag = "Luna16" in image_filename


		image = np.load(os.path.join(self.images_directory, image_filename))
		
		lungs = np.load(os.path.join(self.images_directory.replace('Xray','Lungs'), lungs_name))

		if s_flag:
			# m = image.max()

			# image = image
			# lungs = lungs

			image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

			if self.transform is not None:
				transformed = self.transform(image=image, mask=lungs)
				image = transformed["image"]
				lungs = transformed["mask"]

			image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
			if self.MS:
				image, mc,sc = standardize(image,self.mean_std[0],self.mean_std[1])
			else:
				image, mc,sc = standardize(image,0,0)
			lungs = normal(lungs)

			masks = lungs

		else:
			masks = lungs

		image = np.expand_dims(image, axis=2)
		image = np.transpose(image, (2, 0, 1))
		
		return image, masks

class BXDataset(Dataset):
	def __init__(self, images_filenames, images_directory, transform=None, MS=False):
		self.images_filenames = images_filenames
		self.images_directory = images_directory
		self.transform = transform
		self.mean_std = [0.496, 0.248] #  NIH-train
		self.MS=MS

	def __len__(self):
		return len(self.images_filenames)

	def __getitem__(self, idx):

		image_filename = self.images_filenames[idx]

		image = cv2.imread(os.path.join(self.images_directory, image_filename), cv2.IMREAD_GRAYSCALE)
		
		image=image/255.0

		image = np.float32(image)
        
		if self.transform is not None:
			transformed = self.transform(image=image)
			image = transformed["image"]

		if self.MS:
			image, mc,sc = standardize(image,self.mean_std[0],self.mean_std[1])
		else:
			image, mc,sc = standardize(image,0,0)
		
		image = np.expand_dims(image, axis=2)
		
		image = np.transpose(image, (2, 0, 1))

		return image, image_filename

class CXDataset(Dataset):
	def __init__(self, images_filenames, images_directory, transform=None, MS=False):
		self.images_filenames = images_filenames
		self.images_directory = images_directory
		self.transform = transform
		self.mean_std = [0.574, 0.257] #  Vin-Dr train
		# self.mean_std = [1.015, 0.345] #  Luna16 train
		self.MS=MS

	def __len__(self):
		return len(self.images_filenames)

	def __getitem__(self, idx):

		image_filename = self.images_filenames[idx]

		image = np.load(os.path.join(self.images_directory, image_filename))

		image = np.float32(image)

		if self.transform is not None:
			transformed = self.transform(image=image)
			image = transformed["image"]

		if self.MS:
			image, mc,sc = standardize(image,self.mean_std[0],self.mean_std[1])
		else:
			image, mc,sc = standardize(image,0,0)
		
		image = np.expand_dims(image, axis=2)
		
		image = np.transpose(image, (2, 0, 1))

		return image, image_filename

class RXrayDataset(Dataset):
	def __init__(self, images_filenames, images_directory, transform=None, MS=False):
		self.images_filenames = images_filenames
		self.images_directory = images_directory
		self.transform = transform
		self.mean_std = [0.574, 0.257] #  Vin-Dr train
		self.MS=MS

	def __len__(self):
		return len(self.images_filenames)

	def __getitem__(self, idx):
		image_filename = self.images_filenames[idx]

		Xray = di.dcmread(os.path.join(self.images_directory, image_filename))
		image = Xray.pixel_array
		
		if Xray.PhotometricInterpretation == 'MONOCHROME1':
			image = np.amax(image) - image

		image=image/(Xray.WindowWidth-1)

		image=resize_images(image,1024,1024,image.shape[0],image.shape[1]) ##

		image = np.float32(image)

		if self.transform is not None:
			transformed = self.transform(image=image)
			image = transformed["image"]
			
		if self.MS:
			image, mc,sc = standardize(image,self.mean_std[0],self.mean_std[1])
		else:
			image, mc,sc = standardize(image,0,0)
		
		image = np.expand_dims(image, axis=2)


		image = np.transpose(image, (2, 0, 1))

		return image

def standardize(img, m, s):
	
	if s==0:
		s = img.std()
	if m==0:
		m = img.mean()
	
	img = (img - m)/s
	
	return img, m, s

def unstandardize(img, m, s):
	
	img = img*s + m
	
	return img
	
def imcrop(img):
	
	if len(img.shape) > 2:
		z, y, x = img.shape
		v = min(x,y,z)
	else:
		y, x = img.shape
		v = min(x,y)
			
	
	if len(img.shape) > 2:
		new_x=v
		new_y=v
		new_z=v

		left = int((y - new_y)/2)
		top = int((x - new_x)/2)
		right = int((y + new_y)/2)
		bottom = int((x + new_x)/2)
		z_top = int((z - new_z)/2)
		z_bottom = int((z + new_z)/2)
		
		k = img[z_top:z_bottom,left:right,top:bottom]
	else:
		new_x=v
		new_y=v

		left = int((y - new_y)/2)
		top = int((x - new_x)/2)
		right = int((y + new_y)/2)
		bottom = int((x + new_x)/2)
		
		k = img[left:right,top:bottom]
	
	return k

def resize_images(img,desired_width,desired_height,current_width,current_height): # img is 1024*1024

	width_index=current_width/desired_width
	height_index=current_height/desired_height

	width_factor=1/width_index
	height_factor=1/height_index
	
	img=ndimage.zoom(img,(width_factor,height_factor),order=1)
	
	return img

def normal(img):
	return (img - img.min())/(img.max()-img.min())