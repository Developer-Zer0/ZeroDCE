import pandas as pd
import numpy as np
import glob
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import pickle

from model import Model
from loss import Loss

# Dataset object, which will perform transformations on the images
class ImageDataset(Dataset):

	def __init__(self, data_list, transform=None):

		self.data_list = data_list
		self.transform = transform

	def __len__(self):
		return len(self.data_list)

	# Function which will open image from location and perform transformations before return the image 
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		img_name = self.data_list[idx]
		image = io.imread(img_name)
		if self.transform:
			image = self.transform(image)
		return image

# Function used to rescale the images to desired dimensions
class Rescale(object):

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		image = sample

		h, w = image.shape[:2]
		if isinstance(self.output_size, int):
			if h > w:
				new_h, new_w = self.output_size * h / w, self.output_size
			else:
				new_h, new_w = self.output_size, self.output_size * w / h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)
		img = transform.resize(image, (new_h, new_w))
		return img

# Function which will convert images to tensors
class ToTensor(object):

	def __call__(self, sample):
		image = sample
		image = image.transpose((2, 0, 1))
		return torch.from_numpy(image)

# Train function which will iterate over the dataloader
def train(model, dataloader, optimizer, batch_size, n, image_size):

	for i_batch, sample in enumerate(dataloader):
		optimizer.zero_grad()
		A = model(sample)
		LE = sample
		A_n = A.reshape(batch_size, n, 3, image_size, image_size)
		# Main iterative formula which is applied to enhance the images
		for iter in range(n):
			LE = LE + torch.mul(torch.mul(A_n[:][iter], LE), (torch.ones(LE.shape) - LE))
		# Backward Propogation
		l = Loss()
		loss = l.compute_losses(sample, LE, A, image_size, n)
		loss.backward()
		optimizer.step()
		print('Batch: ', str(i_batch), ' ------ Loss: ', str(loss.data))

if __name__ == '__main__':

	# Fetch image locations from dataset directory
	dataset_directory = 'Dataset'
	image_locs = glob.glob(dataset_directory + '/*.jpg')
	
	train_set = image_locs[:int(0.8*len(image_locs))]

	# Initialize image dataset
	dataset = ImageDataset(train_set, transform=transforms.Compose([Rescale(512),ToTensor()]))

	image_size = 512
	batch_size = 8
	n_epochs = 10
	n = 8
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

	model = Model()
	print('Model created')
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	for epoch in range(0, n_epochs):
		print('Epoch Number: ' + str(epoch))
		train(model.float(), dataloader, optimizer, batch_size, n, image_size)

	# Storing trained model
	pickle.dump(model, open('model/trained_model.pkl', 'wb'))
