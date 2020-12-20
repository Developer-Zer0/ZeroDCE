import torch
import torch.nn as nn

class Loss():
	
	def __init__(self):

		# Predefined filters for calculating spatial loss
		self.w_t = torch.tensor([[[0, -1/3, 0], [0, 1/3, 0], [0, 0, 0]]], dtype=torch.float64).repeat(3,1,1).unsqueeze(0)
		self.w_b = torch.tensor([[[0, 0, 0], [0, 1/3, 0], [0, -1/3, 0]]], dtype=torch.float64).repeat(3,1,1).unsqueeze(0)
		self.w_l = torch.tensor([[[0, 0, 0], [-1/3, 1/3, 0], [0, 0, 0]]], dtype=torch.float64).repeat(3,1,1).unsqueeze(0)
		self.w_r = torch.tensor([[[0, 0, 0], [0, 1/3, -1/3], [0, 0, 0]]], dtype=torch.float64).repeat(3,1,1).unsqueeze(0)
		self.weights_spatial = torch.cat([self.w_t, self.w_b, self.w_l, self.w_r], dim=0).squeeze(0)

		# Predefined filters for calculating illumination loss
		self.w_h = torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], dtype=torch.float32).repeat(24,1,1,1)
		self.w_v = torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]], dtype=torch.float32).repeat(24,1,1,1)
		
		# Average Pooling layers required in calculating losses
		self.avg_pool_4 = nn.AvgPool2d(4, stride=4)
		self.avg_pool_16 = nn.AvgPool2d(16, stride=16)

	# Function which calls all of the loss functions and returns the total loss
	def compute_losses(self, input, output, A, size, n):

		w_col = 0.5
		w_tva = 20
		
		l_spa = self.spatial_loss(input, output)
		l_exp = self.exposure_loss(output)
		l_col = self.color_loss(output)
		l_tva = self.illumination_loss(A, size, n)

		return l_spa + l_exp + w_col*l_col + w_tva*l_tva

	def spatial_loss(self, i, o):

		i = self.avg_pool_4(i)
		o = self.avg_pool_4(o)
		d_i = nn.functional.conv2d(i, self.weights_spatial, padding=1, stride=1)
		d_o = nn.functional.conv2d(o, self.weights_spatial, padding=1, stride=1)
		d = torch.square(torch.abs(d_o) - torch.abs(d_i))
		s = torch.sum(d,dim=1)
		l_spa = torch.mean(s)
		return l_spa

	def exposure_loss(self, o):

		E = 0.6
		o = self.avg_pool_16(o)
		o = torch.abs(o - E*torch.ones(o.shape))
		l_exp = torch.mean(o)
		return l_exp

	def color_loss(self, o):

		avg_intensity_channel = torch.mean(o, dim=(2,3))
		avg_intensity_channel_rolled = torch.roll(avg_intensity_channel, 1, 1)
		d_j = torch.square(torch.abs(avg_intensity_channel - avg_intensity_channel_rolled))
		l_col = torch.mean(torch.sum(d_j, dim=1))
		return l_col

	def illumination_loss(self, A, size, n):

		h_grad = nn.functional.conv2d(A, self.w_h, padding=1, groups=n*3)
		v_grad = nn.functional.conv2d(A, self.w_v, padding=1, groups=n*3)
		h_grad = h_grad.reshape(-1, n, 3, size, size)
		v_grad = v_grad.reshape(-1, n, 3, size, size)
		h_grad = torch.mean(h_grad, dim=(3,4))
		v_grad = torch.mean(v_grad, dim=(3,4))
		grad = torch.square(torch.abs(h_grad) + torch.abs(v_grad))
		grad = torch.sum(grad, dim=2)
		l_tva = torch.mean(grad)
		return l_tva