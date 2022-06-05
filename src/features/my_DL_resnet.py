import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features.layer_functions import Res_block, reshape_to_voxel

class ResNet_DL(nn.Module):
    def __init__(
        self, 
        n_class, 
        n_height, 
        circular_padding, 
        grid_size
        ):
        super(ResNet_DL, self).__init__()
        self.circular_padding = circular_padding
        self.n_class= n_class
        self.n_height = n_height

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_height, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
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
        self.ASPP_module = ASPP(2048, 32, 256)
        self.up1 = nn.Upsample(size=(grid_size[0]//4, grid_size[1]//4), mode= 'bilinear')
        self.concat = nn.Conv2d(512, 48, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(304,256,3,1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 32, 1)
        )
        self.up2 = nn.Upsample(size=(grid_size[0], grid_size[1]), mode= 'bilinear')
        self.classifier = nn.Conv2d(32, n_class*n_height, kernel_size=1)

    def forward(self,x):                        # 2, 32, 480, 360
        x1 = self.conv1(x)                      # 2, 64, 240, 180
        x2 = self.conv2_x(x1)                   # 2, 256, 240, 180
        x3 = self.conv3_x(x2)                   # 2, 512, 120, 90
        x4 = self.conv4_x(x3)                   # 2, 1024, 60,45
        x5 = self.conv5_x(x4)                   # 2, 2048, 30,23
        out1 = self.up1(self.ASPP_module(x5))   # 2, 256, 120, 90
        out2 = self.concat(x3)                  # 2, 48, 120, 90
        x = torch.cat([out1, out2], 1)          # 2, 304, 120, 90
        x = self.up2(self.conv(x))
        x = self.classifier(x)
        x = reshape_to_voxel(x, self.n_height, self.n_class)
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, concat_channels):
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
        self.conv2 = nn.Conv2d(out_channels, concat_channels, kernel_size=1)

    def forward(self, x):
        x_h, x_w = x.size()[2], x.size()[3]

        out1, out2, out3, out4, out5 = self.ASPP1(x), self.ASPP2(x), self.ASPP3(x), self.ASPP4(x), self.ASPP5(x)
        out5 = F.upsample(out5, size=(x_h, x_w), mode="bilinear")

        out = torch.cat([out1, out2, out3, out4, out5], 1) 
        out = self.conv(out)
        out = self.conv2(out) 

        return out
