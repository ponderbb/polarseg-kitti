import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import double_CBR, down_CBR, up_CBR

class BEV_Unet(nn.Module):

    def __init__(self,n_class,n_height,dropout = 0.5,circular_padding = False, block_size = 7):
        super(BEV_Unet, self).__init__()
        self.n_class = n_class
        self.n_height = n_height
        self.Unet = UNet(n_class*n_height, n_height, dropout,circular_padding, block_size)

    def forward(self, x, circular_padding):
        x = self.Unet(x, circular_padding)
        x = x.permute(0,2,3,1)
        new_shape = list(x.size())[:3] + [self.n_height, self.n_class]
        x = x.view(new_shape)
        x = x.permute(0,4,1,2,3)
        return x


class UNet(nn.Module):

    def __init__(self, n_class,n_height, dropout,circular_padding, block_size):
        super(UNet, self).__init__()
        self.norm = nn.BatchNorm2d(n_height)
        self.inc = double_CBR(n_height, 64, circular_padding, 3)
        self.down1 = down_CBR(64, 128, circular_padding, 3)
        self.down2 = down_CBR(128, 256, circular_padding, 3)
        self.down3 = down_CBR(256, 512, circular_padding, 3)
        self.down4 = down_CBR(512, 512, circular_padding, 3)
        self.up1 = up_CBR(1024, 256, circular_padding, 3, drop_p=dropout, block_size = block_size)
        self.up2 = up_CBR(512, 128, circular_padding, 3, drop_p=dropout, block_size = block_size)
        self.up3 = up_CBR(256, 64, circular_padding, 3, drop_p=dropout, block_size = block_size)
        self.up4 = up_CBR(128, 64, circular_padding, 3, drop_p=dropout, block_size = block_size)
        self.outc = nn.Conv2d(64, n_class, 1)

    def forward(self, x, circular_padding):
        x = self.norm(x)
        x1 = self.inc(x, circular_padding)
        x2 = self.down1(x1, circular_padding)
        x3 = self.down2(x2, circular_padding)
        x4 = self.down3(x3, circular_padding)
        x5 = self.down4(x4, circular_padding)
        x = self.up1(x5, x4, circular_padding)
        x = self.up2(x, x3, circular_padding)
        x = self.up3(x, x2, circular_padding)
        x = self.up4(x, x1, circular_padding)
        x = self.outc(x)
        return x
