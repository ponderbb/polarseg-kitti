import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D


class BEV_Unet(nn.Module):

    def __init__(self,n_class,n_height,dropout = 0.,circular_padding = False, dropblock = True):
        super(BEV_Unet, self).__init__()
        self.n_class = n_class
        self.n_height = n_height
        self.network = UNet(n_class*n_height,n_height,dropout,circular_padding,dropblock)

    def forward(self, x, circular_padding):
        x = self.network(x, circular_padding)
        
        x = x.permute(0,2,3,1)
        new_shape = list(x.size())[:3] + [self.n_height,self.n_class]
        x = x.view(new_shape)
        x = x.permute(0,4,1,2,3)
        return x


class UNet(nn.Module):
    def __init__(self, n_class,n_height, dropout,circular_padding,dropblock):
        super(UNet, self).__init__()

        self.inc = inconv(n_height, 64, circular_padding)
        self.down1 = down(64, 128, circular_padding)
        self.down2 = down(128, 256, circular_padding)
        self.down3 = down(256, 512, circular_padding)
        self.down4 = down(512, 512, circular_padding)
        self.up1 = up(1024, 256, circular_padding, use_dropblock=dropblock, drop_p=dropout)
        self.up2 = up(512, 128, circular_padding, use_dropblock=dropblock, drop_p=dropout)
        self.up3 = up(256, 64, circular_padding, use_dropblock=dropblock, drop_p=dropout)
        self.up4 = up(128, 64, circular_padding, use_dropblock=dropblock, drop_p=dropout)
        self.dropout = nn.Dropout(p=0. if dropblock else dropout)
        self.outc = outconv(64, n_class)

    def forward(self, x, circular_padding):
        x1 = self.inc(x, circular_padding)
        x2 = self.down1(x1, circular_padding)
        x3 = self.down2(x2, circular_padding)
        x4 = self.down3(x3, circular_padding)
        x5 = self.down4(x4, circular_padding)
        x = self.up1(x5, x4, circular_padding)
        x = self.up2(x, x3, circular_padding)
        x = self.up3(x, x2, circular_padding)
        x = self.up4(x, x1, circular_padding)
        x = self.outc(self.dropout(x))
        return x

class double_CBR(nn.Module):
    def __init__(self, in_ch, out_ch, circular_padding):
        super(double_CBR, self).__init__()
        if circular_padding :
            padding_version = (1,0)
        else : 
            padding_version = 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=padding_version),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=padding_version),
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


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, circular_padding):
        super(inconv, self).__init__()
        self.norm = nn.BatchNorm2d(in_ch)
        self.conv = double_CBR(in_ch, out_ch, circular_padding)

    def forward(self, x, circular_padding):
        x = self.norm(x)
        x = self.conv(x, circular_padding)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, circular_padding):
        super(down, self).__init__()
        self.pooling = nn.MaxPool2d(2)
        self.conv = double_CBR(in_ch, out_ch, circular_padding)
      
    def forward(self, x, circular_padding):
        x = self.pooling(x)
        x = self.conv(x, circular_padding)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, circular_padding, bilinear=True, use_dropblock = False, drop_p = 0.5):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = double_CBR(in_ch, out_ch, circular_padding)

        self.use_dropblock = use_dropblock

        if self.use_dropblock:
            self.dropblock = DropBlock2D(block_size=7, drop_prob=drop_p)

    def forward(self, x1, x2, circular_padding):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x, circular_padding)
        if self.use_dropblock:
            x = self.dropblock(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x