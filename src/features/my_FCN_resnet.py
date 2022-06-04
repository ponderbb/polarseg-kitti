import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features.layer_functions import Res_block, up_CBR, CBR, reshape_to_voxel

class ResNet_FCN(nn.Module):
    def __init__(self, n_class, n_height, circular_padding):
        super(ResNet_FCN, self).__init__()
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
        layers2 = [Res_block(i, m , o, circular_padding) for i, m , o in zip(conv2_in_ch, conv2_mid_ch, conv2_out_ch)]
        self.conv2_x = nn.Sequential(*layers2)

        conv3_in_ch, conv3_mid_ch, conv3_out_ch = [256]+[512 for i in range(3)], [128 for i in range(4)], [512 for i in range(4)]
        layers3 = [Res_block(i, m , o, circular_padding) for i, m , o in zip(conv3_in_ch, conv3_mid_ch, conv3_out_ch)]
        self.conv3_x = nn.Sequential(*layers3)

        conv4_in_ch, conv4_mid_ch, conv4_out_ch = [512]+[1024 for i in range(5)], [256 for i in range(6)], [1024 for i in range(6)]
        layers4 = [Res_block(i, m , o, circular_padding) for i, m , o in zip(conv4_in_ch, conv4_mid_ch, conv4_out_ch)]
        self.conv4_x = nn.Sequential(*layers4)

        conv5_in_ch, conv5_mid_ch, conv5_out_ch = [1024,2048,2048], [512 for i in range(3)], [2048 for i in range(3)]
        layers5 = [Res_block(i, m , o, circular_padding) for i, m , o in zip(conv5_in_ch, conv5_mid_ch, conv5_out_ch)]
        self.conv5_x = nn.Sequential(*layers5)

        self.up1 = up_CBR(4096, 1024, self.circular_padding, "FCN")
        self.up2 = up_CBR(2048,512, self.circular_padding,"FCN")
        self.up3 = up_CBR(1024,  256, self.circular_padding, "FCN")
        self.up4 = up_CBR( 512, 64, self.circular_padding,  "FCN")
        self.up5 = up_CBR(128, 32, self.circular_padding, "UNet")
        self.up6 = nn.ConvTranspose2d(32, 32, 2, stride=2, groups=32)
        self.classifier = nn.Conv2d(32, n_class*n_height, kernel_size=1)

    def forward(self,x):                        # 2, 32, 480, 360
        x0 = self.conv1(x)                      # 2, 64, 240, 180
        x1 = self.maxpool1(x0)                  # 2, 64, 120, 90
        x2 = self.conv2_x(x1)                   # 2, 256, 120, 90
        x3 = self.conv3_x(x2)                   # 2, 512, 60, 45
        x4 = self.conv4_x(x3)                   # 2, 1024, 30, 23
        x5 = self.conv5_x(x4)                   # 2, 2048, 15, 12
        score5 = self.up1(x5,x4)                # 2, 1024, 30, 23
        score4 = self.up2(score5, x3)           # 2, 512, 60, 45
        score3 = self.up3(score4, x2)           # 2, 256, 120, 90
        score2 = self.up4(score3, x1)           # 2, 64, 120, 90
        score1 = self.up5(score2, x0)           # 2, 32, 240, 180
        score = self.up6(score1)                # 2, 32, 480, 360
        x = self.classifier(score)              # 2, 32*19, 480, 360
        x = reshape_to_voxel(x, self.n_height, self.n_class)
        return x
