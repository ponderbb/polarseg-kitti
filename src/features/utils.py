import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class double_CBR(nn.Module):
    def __init__(self, in_ch, out_ch, circular_padding, kernel_size):
        super(double_CBR, self).__init__()
        if circular_padding :
            padding_version = (1,0)
        else : 
            padding_version = 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding_version),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding_version),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, circular_padding):
        if circular_padding : 
            x = F.pad(x,(1,1,0,0),mode = 'circular')
            x = self.conv1(x)
            x = F.pad(x,(1,1,0,0),mode = 'circular')
            x = self.conv2(x)
        else : 
            x = self.conv1(x)
            x = self.conv2(x)
        return x

class down_CBR(nn.Module):
    def __init__(self, in_ch, out_ch, circular_padding, kernel_size):
        super(down_CBR, self).__init__()
        self.pooling = nn.MaxPool2d(2)
        self.conv = double_CBR(in_ch, out_ch, circular_padding, kernel_size)
      
    def forward(self, x, circular_padding):
        x = self.pooling(x)
        x = self.conv(x, circular_padding)
        return x


class up_CBR(nn.Module):
    def __init__(self, in_ch, out_ch, circular_padding, kernel_size, drop_p = 0.5, block_size = 7):
        super(up_CBR, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = double_CBR(in_ch, out_ch, circular_padding, kernel_size)
        self.drop_prob = drop_p
        self.block_size = block_size

    def forward(self, x1, x2, circular_padding):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x, circular_padding)

        gamma = self.drop_prob / (self.block_size ** 2)

        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float().to(x.device)

        block_mask = compute_block_mask(mask, 7)

        out = x * block_mask[:, None, :, :]
        out = out * block_mask.numel() / block_mask.sum()

        return out


def compute_block_mask(mask, block_size):
    block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                kernel_size=(block_size, block_size),
                                stride=(1, 1),
                                padding=block_size // 2)

    if block_size % 2 == 0:
        block_mask = block_mask[:, :, :-1, :-1]

    block_mask = 1 - block_mask.squeeze(1)

    return block_mask
    

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )
            
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x