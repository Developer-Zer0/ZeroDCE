import torch
import torch.nn as nn

# Main CNN model
class Model(nn.Module):

	def __init__(self):

		super(Model, self).__init__()

		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()

		self.conv3_32 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
		self.conv32_32 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.conv64_32 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
		self.conv64_24 = nn.Conv2d(64, 24, kernel_size=3, stride=1, padding=1)

	# Total of 7 layers with skip connections
	def forward(self, inpt):

		output1 = self.relu(self.conv3_32(inpt.float()))
		output2 = self.relu(self.conv32_32(output1))
		output3 = self.relu(self.conv32_32(output2))
		output4 = self.relu(self.conv32_32(output3))

		output5 = self.relu(self.conv64_32(torch.cat([output4, output3], dim=1)))
		output6 = self.relu(self.conv64_32(torch.cat([output5, output2], dim=1)))

		output7 = self.tanh(self.conv64_24(torch.cat([output6, output1], dim=1)))

		return output7