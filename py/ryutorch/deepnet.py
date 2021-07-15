
import torch
import torch.nn as nn
from torch import optim


class MLP(nn.Module):
    def __init__(self, num_class, input_channel,  hidden_dim):
        super(MLP, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_channel, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)


class VGG(nn.Module):
    def __init__(
        self,
        # 特征提取层
        num_cls: int = 1000,
        output_channel=512,
        input_channel=3
    ) -> None:
        super(VGG, self).__init__()

        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(self.output_channel[0], self.output_channel[0], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(self.output_channel[1], self.output_channel[1], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(
                self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),
            nn.Conv2d(
                self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),
        )
        self.avgpool=nn.AdaptiveAvgPool2d((8,8))
        self.classifier = nn.Sequential(
            nn.Linear(self.output_channel[3] * 8 * 8, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_cls),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def getNetwork(name, num_cls: int, input_dim=None):
    if name=="vgg":
        return VGG(num_cls=num_cls),(64,64)

def getOptimizer(net, name, **kwargs):

    if name == "adam":
        optimizer = optim.Adam(net.parameters(), **kwargs)
    elif name=="sgd":
        optimizer=optim.SGD(net.parameters(),**kwargs)

    return optimizer

def getLoss(name):

    if name=="crossentropy":
        return nn.CrossEntropyLoss()

