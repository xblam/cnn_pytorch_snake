import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from game import SnakeGameAI

DEVICE =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # in channels set to 2 because there is only red and white
        # size of the filter kernel will be 3x3
        # we will have 32 filters
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=1, padding=1)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256*8*8, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        # put the data through the convolutional layer, activate it using relu, and then pool the activations into pooling layer
        x = torch.relu(self.conv1(x))
        x = x.view(-1, 256*8*8)  # Flatten the tensor
        x = torch.relu(self.fc1(x)) # apply relu activation function of the tensor
        x = self.fc2(x)
        return x # x is a vector of three final activation values, we just have to pick out the highest one

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainerCNN:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()


    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).to(DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float).to(DEVICE)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 3:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

if __name__ == '__main__':
    env = SnakeGameAI(True)
    model = SimpleCNN().to(DEVICE)

    # Create a dummy input tensor
    # Shape: [batch_size, in_channels, height, width]
    import numpy as np
    game_matrix = env.game_matrix
    input_tensor = torch.tensor(game_matrix, dtype=torch.float).to(DEVICE)

    # Pass the input through the model
    print(input_tensor)
    output = model(input_tensor)

    # Print resulting predictions
    print(output)