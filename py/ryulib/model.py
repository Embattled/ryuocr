import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class):
        super(MLP, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        #self.bn3 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(hidden_dim, num_class)

        # Activation func.
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_input):

        # flatten the tensor x -> 800
        b, c, h, w = x_input.size()

        x = x_input.view(b, -1)

        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        # batch, channels, height, width
        return x
