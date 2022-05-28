import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features.utils import BottleNeck, up_CBR


# TODO Add decoder part
class DeepLab_ResNet(nn.Module):
    def __init__(self, num_block, n_class, n_height, circular_padding):
        super(DeepLab_ResNet, self).__init__()

        self.circular_padding = circular_padding
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_height, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.conv2_x = self._make_layer(64, num_block[0], 1)
        self.conv3_x = self._make_layer(128, num_block[1], 2)
        self.conv4_x = self._make_layer(256, num_block[2], 2)
        self.conv5_x = self._make_layer(512, num_block[3], 2)

        

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BottleNeck(self.in_channels, out_channels, self.circular_padding, stride, self.circular_padding))
            self.in_channels = out_channels * BottleNeck.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2_x(x1)
        x3 = self.conv3_x(x2)
        x4 = self.conv4_x(x3)
        x5 = self.conv5_x(x4)

        

        return score
