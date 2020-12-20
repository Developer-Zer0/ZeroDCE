import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import pickle
from random import randrange

from model import Model

# image_locs = glob.glob('/content/drive/MyDrive/Datasets/ZeroDCE' + '/*.jpg')

class ImageDataset(Dataset):

	def __init__(self, data_list, transform=None):

		self.data_list = data_list
		self.transform = transform

	def __len__(self):
		return len(self.data_list)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		img_name = self.data_list[idx]
		image = io.imread(img_name)
		if self.transform:
			image = self.transform(image)
		return image

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

class ToTensor(object):

	def __call__(self, sample):
		image = sample
		image = image.transpose((2, 0, 1))
		return torch.from_numpy(image)

# Fetch image locations from dataset directory
dataset_directory = 'Dataset'
image_locs = glob.glob(dataset_directory + '/*.jpg')	
test_set = image_locs[int(0.8*len(image_locs)):]
test_image = test_set[randrange(len(test_set))]

dataset = ImageDataset([test_image], transform=transforms.Compose([Rescale(512), ToTensor(),]))

image_size = 512
batch_size = 8
n_epochs = 10
n = 8
model_path = 'model/model_dce_2.pkl'
model = pickle.load(open(model_path, 'rb'))

img = dataset[0]
fig1, ax1 = plt.subplots()
ax1.imshow(img.permute(1, 2, 0))
img = img.unsqueeze(0)
a = model(img)
a_n = a.reshape(1, n, 3, image_size, image_size)
LE = img
for iter in range(n):
		LE = LE + torch.mul(torch.mul(a_n[0][iter], LE), (torch.ones(LE.shape) - LE))
fig2, ax2 = plt.subplots()
ax2.imshow(LE.squeeze().cpu().detach().permute(1, 2, 0))

plt.show()