import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils


from skimage import feature


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class):
        super(MLP, self).__init__()

        # Fully connected layers
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, num_class)

        # self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        # self.sigmoid = nn.Sigmoid()

        self.feature=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.Tanh(),
            # nn.ReLU(inplace=True),
            nn.Linear(hidden_dim,num_class),
            nn.Sigmoid()
        )

    def forward(self, x):

        # x = self.relu(self.fc1(x))
        # x = self.tanh(self.fc1(x))
        # x = self.sigmoid(self.fc2(x))

        return self.feature(x)
