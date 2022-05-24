import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import BottleNeck
from torchvision.models.vgg import VGG

#TODO Need to add circular padding
class FCN_ResNet(nn.Module):
    def __init__(self, block, num_block, n_class, n_height,):
        super(FCN_ResNet, self).__init__()

        self.in_channels=64

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_height, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1) 


    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2_x(x1)
        x3 = self.conv3_x(x2)
        x4 = self.conv4_x(x3)
        x5 = self.conv5_x(x4)
        
        score = self.bn1(self.relu(self.deconv1(x5)))     
        score = score + x4                                
        score = self.bn2(self.relu(self.deconv2(score)))  
        score = score + x3                                
        score = self.bn3(self.relu(self.deconv3(score)))  
        score = score + x2                                
        score = self.bn4(self.relu(self.deconv4(score)))  
        score = score + x1                                
        score = self.bn5(self.relu(self.deconv5(score)))  
        score = self.classifier(score)                    

        return score  

def resnet50():
    return FCN_ResNet(BottleNeck, [3,4,6,3])

def resnet101():
    return FCN_ResNet(BottleNeck, [3, 4, 23, 3])

