from tkinter import X
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features.layer_functions import Res_block, up_CBR, CBR

class ResNet_DL(nn.Module):
    def __init__(self, n_class, n_height, circular_padding, grid_size):
        super(ResNet_DL, self).__init__()
        self.circular_padding = circular_padding
        self.n_class= n_class
        self.n_height = n_height

        self.conv1 = CBR(in_ch=n_height, out_ch=64, circular_padding=False, filter_size=7, stride=2, padding_size=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        conv2_in_ch, conv2_mid_ch, conv2_out_ch = [64, 256, 256], [64 for i in range(3)], [256 for i in range(3)]
        layers2 = [Res_block(i, m , o, self.circular_padding) for i, m , o in zip(conv2_in_ch, conv2_mid_ch, conv2_out_ch)]
        self.conv2_x = nn.Sequential(*layers2)

        conv3_in_ch, conv3_mid_ch, conv3_out_ch = [256]+[512 for i in range(3)], [128 for i in range(4)], [512 for i in range(4)]
        layers3 = [Res_block(i, m , o, self.circular_padding) for i, m , o in zip(conv3_in_ch, conv3_mid_ch, conv3_out_ch)]
        self.conv3_x = nn.Sequential(*layers3)

        conv4_in_ch, conv4_mid_ch, conv4_out_ch = [512]+[1024 for i in range(22)], [256 for i in range(23)], [1024 for i in range(23)]
        layers4 = [Res_block(i, m , o, self.circular_padding) for i, m , o in zip(conv4_in_ch, conv4_mid_ch, conv4_out_ch)]
        self.conv4_x = nn.Sequential(*layers4)

        conv5_in_ch, conv5_mid_ch, conv5_out_ch = [1024,2048,2048], [512 for i in range(3)], [2048 for i in range(3)]
        layers5 = [Res_block(i, m , o, self.circular_padding) for i, m , o in zip(conv5_in_ch, conv5_mid_ch, conv5_out_ch)]
        self.conv5_x = nn.Sequential(*layers5)
        self.ASPP_module = ASPP(2048, 32, self.n_class*self.n_height)
        self.up = nn.Upsample(size=(grid_size[0], grid_size[1]), mode= 'bilinear')


    def forward(self,x):                        # 2, 32, 480, 360
        x1 = self.conv1(x)                      # 2, 64, 240, 180
        x2 = self.conv2_x(x1)                   # 2, 256, 240, 180
        x3 = self.conv3_x(x2)                   # 2, 512, 120, 90
        x4 = self.conv4_x(x3)                   # 2, 1024, 60,45
        x5 = self.conv5_x(x4)                   # 2, 2048, 30,23
        x = self.ASPP_module(x5)                # 2, 608, 30, 23
        x = self.up(x)                          # 2, 608, 480, 360
        
        x = x.permute(0, 2, 3, 1)
        class_per_voxel_dim = [x.size()[0], x.size()[1], x.size()[2], self.n_height, self.n_class]
        x = x.view(class_per_voxel_dim).permute(0, 4, 1, 2, 3)

        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(ASPP, self).__init__()
        
        self.ASPP1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.ASPP2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.ASPP3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.ASPP4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.ASPP5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.classifier = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x_h, x_w = x.size()[2], x.size()[3]

        out1, out2, out3, out4, out5 = self.ASPP1(x), self.ASPP2(x), self.ASPP3(x), self.ASPP4(x), self.ASPP5(x)
        out5 = F.upsample(out5, size=(x_h, x_w), mode="bilinear")

        out = torch.cat([out1, out2, out3, out4, out5], 1) 
        out = self.conv(out)
        out = self.classifier(out) 

        return out
