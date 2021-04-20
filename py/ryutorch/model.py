import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class):
        super(MLP, self).__init__()

        self.classifier=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim,num_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)
