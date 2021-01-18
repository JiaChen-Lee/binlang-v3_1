import torch
import torch.nn as nn


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.conv1x1_1 = nn.Conv2d()


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=7,
                               stride=2,
                               padding=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3,
                                      stride=2)
        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=192,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3,
                                      stride=2)
