import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

DEVICE =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # in channels set to 2 because there is only red and white
        # size of the filter kernel will be 3x3
        # we will have 32 filters
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32*4*4, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        # put the data through the convolutional layer, activate it using relu, and then pool the activations into pooling layer
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32*4*4)  # Flatten the tensor
        x = torch.relu(self.fc1(x)) # apply relu activation function of the tensor
        x = self.fc2(x)
        return x # x is a vector of three final activation values, we just have to pick out the highest one

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)