
import torch
import torch.nn as nn
from torch import optim
from torchvision import models


def make_classifier(input_dim, n_classes, arch: str, p_dropout: float = 0.5, n_linear: int = 4096):
    classifier = []
    for i in range(len(arch)):
        if arch[i] == 'L':
            classifier.extend([
                nn.Linear(input_dim, n_linear),
                nn.ReLU(inplace=True)
            ])
            input_dim = n_linear
        elif arch[i] == 'D':
            classifier.append(nn.Dropout(p_dropout))
        else:
            raise ValueError("Illegal classifier architecture '{}'." % (arch))
    classifier.append(nn.Linear(input_dim, n_classes))
    return nn.Sequential(*classifier)


class MLP(nn.Module):
    def __init__(self, num_class, input_channel,  hidden_dim):
        super(MLP, self).__init__()

        # self.classifier = nn.Sequential(
        #     nn.Linear(input_channel, hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, num_class),
        #     nn.Sigmoid()
        # )
        self.classifier = make_classifier(
            input_channel, num_class, "L", n_linear=hidden_dim)

    def forward(self, x):
        return self.classifier(x)


class MyVGG(nn.Module):
    def __init__(
        self,
        # number of convolution layers per down sampling
        numlayer_perds=2,
        num_classes: int = 1000,
        classifier_arch: str = "LDLD",

        feature_size: int = 4,
        feature_channel: int = 512,
        channel_time: int = 2,

        input_size: int = 64,
        input_channel: int = 3,

        batch_norm=False,
        **kwargs
    ) -> None:
        super(MyVGG, self).__init__()

        self.input_size = input_size
        self.features_size = feature_channel * feature_size * feature_size
        self.features = self.make_feature(
            numlayer_perds, feature_size, feature_channel, channel_time, input_size, input_channel, batch_norm)

        self.avgpool = nn.AdaptiveAvgPool2d((feature_size, feature_size))
        self.classifier = make_classifier(
            self.features_size, num_classes, classifier_arch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def make_feature(self, numlayer_perds, feature_size, feature_channel, channel_time, input_size, input_channel: int, batch_norm):
        from math import log2
        num_dp = int(log2(input_size/feature_size))
        if isinstance(numlayer_perds, int):
            numlayer_perds = [numlayer_perds]*num_dp
        elif not isinstance(numlayer_perds, list) or len(numlayer_perds) != num_dp:
            raise ValueError(
                "numlayer_perds must is int or list with same length as log2(input_size/feature_size)")

        num_channels = []
        for i in range(num_dp):
            channel = feature_channel//(channel_time**(num_dp-i-1))
            if channel == 0:
                raise ValueError(
                    "Illegal parameter, calculated output channel of first conv is 0!")
            num_channels.append(channel)

        layers: List[nn.Module] = []
        for dp in range(num_dp):
            out_channel = num_channels[dp]
            for i in range(numlayer_perds[dp]):
                conv2d = nn.Conv2d(input_channel, out_channel,
                                   kernel_size=3, padding=1)
                input_channel = out_channel
                if batch_norm == True:
                    layers += [conv2d,
                               nn.BatchNorm2d(out_channel), nn.ReLU(True)]
                else:
                    layers += [conv2d, nn.ReLU(True)]

            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)


class MyNonePoolNet(nn.Module):
    def __init__(
        self,
        kernel_size: int or list(int),
        stride: int = 2,
        num_classes: int = 1000,
        feature_size: int = 4,
        classifier_arch: str = "LDLD",

        feature_channel: int = 512,
        input_size: int = 64,
        input_channel: int = 3,
        first33: bool = True,
        batch_norm=False,
        **kwargs

    ) -> None:
        super(MyNonePoolNet, self).__init__()

        self.input_size = input_size
        self.features_size = feature_channel * feature_size * feature_size

        self.features = self.make_feature(
            kernel_size, stride, feature_size, feature_channel, input_size, input_channel, first33, batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((feature_size, feature_size))
        self.classifier = make_classifier(
            self.features_size, num_classes, classifier_arch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def make_feature(self, kernel_size, stride, feature_size, feature_channel, input_size, input_channel: int, first33: bool, batch_norm):
        from math import ceil, log2, log
        num_layers = log(input_size/feature_size, stride)

        # check num_layers
        if num_layers != int(num_layers):
            raise ValueError("Cant create MyNet auto with: in %d fe %d stride %d" % (
                input_size, feature_size, stride))
        num_layers = int(num_layers)

        # check kernel size
        if isinstance(kernel_size, int):
            if kernel_size < stride:
                raise ValueError(
                    "Create MyNet: dim of kernel can't less than stride.")
            num_padding = [ceil((kernel_size-stride)/2)]*num_layers
            kernel_size = [kernel_size]*num_layers
        elif isinstance(kernel_size, list):
            if len(kernel_size) != num_layers or any(x < stride for x in kernel_size):
                raise ValueError(
                    "Create Mynet: If kernel size is list, must have same length with num of layers, and all size should not less than stride.")
            num_padding = [ceil((x-stride)/2) for x in kernel_size]
        else:
            raise ValueError("Kernel size must is int or list of int")

        layers: List[nn.Module] = []
        num_channels = []
        for i in range(num_layers):
            num_channels.append(feature_channel//(2**(num_layers-i)))
        num_channels.append(feature_channel)

        if first33 == True:
            if batch_norm == True:
                layers += [nn.Conv2d(input_channel, num_channels[0],
                                     3, 1, 1),
                           nn.BatchNorm2d(num_channels[0]), nn.ReLU(True)]
            else:
                layers += [nn.Conv2d(input_channel, num_channels[0],
                                     3, 1, 1), nn.ReLU(True)]
        else:
            num_channels[0] = input_channel

        for i in range(num_layers):
            conv2d = nn.Conv2d(
                num_channels[i], num_channels[i+1], kernel_size[i], stride, num_padding[i])

            if batch_norm == True:
                layers += [conv2d,
                           nn.BatchNorm2d(num_channels[i+1]), nn.ReLU(True)]
            else:
                layers += [conv2d, nn.ReLU(True)]
        return nn.Sequential(*layers)


def getNetwork(name, num_classes: int, parameter: dict = None):
    if name == "myvgg":
        net = MyVGG(num_classes=num_classes, **parameter)
        return net, (net.input_size, net.input_size)
    elif name == "mynonepoolnet":
        net = MyNonePoolNet(num_classes=num_classes, **parameter)
        return net, (net.input_size, net.input_size)
    elif name == "vgg11_offical":
        return models.vgg11(num_classes=num_classes), (224, 224)
    elif name == "alexnet_offical":
        return models.alexnet(num_classes=num_classes), (224, 224)
    elif name == "resnet34_offical":
        return models.resnet34(num_classes=num_classes), (224, 224)
    elif name == "mobilenet_offical":
        return models.MobileNetV2(num_classes), (224, 224)
    else:
        raise ValueError("Unsupport network name :{}".format(name))


def getOptimSched(optimizer, name, endepoch=0, **kwargs):
    if name == "exponential":
        return optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif name == "lineral":
        if endepoch == 0:
            raise ValueError("endepoch parameter is necessary.")

        def liner_func(epoch): return max(0, 1-(epoch/endepoch))
        return optim.lr_scheduler.LambdaLR(optimizer, liner_func, **kwargs)
    else:
        raise ValueError("Unsupport network name :{}".format(name))


def getOptimizer(net, name, scheduler: dict = None, **kwargs):

    if name == "adam":
        optimizer = optim.Adam(net.parameters(), **kwargs)
    elif name == "sgd":
        optimizer = optim.SGD(net.parameters(), **kwargs)

    lr_sched = None
    if scheduler != None:
        lr_sched = getOptimSched(optimizer, **scheduler)
    return optimizer, lr_sched


def getLoss(name):

    if name == "crossentropy":
        return nn.CrossEntropyLoss()
