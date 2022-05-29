import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D

from src.features.utils import down_CBR, up_CBR, CBR, BlockDrop

class BEV_Unet(nn.Module):

    def __init__(self,n_class,n_height,circular_padding = False):
        super(BEV_Unet, self).__init__()
        self.n_class = n_class
        self.n_height = n_height
        self.backbone = UNet(n_class*n_height,n_height,circular_padding)

    def forward(self, x):
        class_per_voxel_dim = [x.size()[0], x.size()[1], x.size()[2], self.n_height, self.n_class]
        x = self.backbone(x)
        x = x.permute(0,2,3,1)
        x = x.view(class_per_voxel_dim).permute(0,4,1,2,3)
        return x
    
class UNet(nn.Module):
    def __init__(self, n_class,n_height,circular_padding):
        super(UNet, self).__init__()
        self.circular_padding=circular_padding
        self.dropout = BlockDrop(drop_p=0.5, block_size=7)
        self.norm = nn.BatchNorm2d(n_height)
        self.inc1 = CBR(n_height, 64, self.circular_padding)
        self.inc2 = CBR(64, 64, self.circular_padding)

        self.down1 = down_CBR(64, 128, self.circular_padding)
        self.down2 = down_CBR(128, 256, self.circular_padding)
        self.down3 = down_CBR(256, 512, self.circular_padding)
        self.down4 = down_CBR(512, 512, self.circular_padding)
        self.conv0 = CBR(512, 512, self.circular_padding)

        self.up1 = up_CBR(1024, 256, self.circular_padding)
        self.conv1 = CBR(256, 256, self.circular_padding)
        self.up2 = up_CBR(512, 128, self.circular_padding)
        self.conv2 = CBR(128, 128, self.circular_padding)
        self.up3 = up_CBR(256, 64, self.circular_padding)
        self.conv3 = CBR(64, 64, self.circular_padding)
        self.up4 = up_CBR(128, 64, self.circular_padding)

        self.outc = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        x = self.norm(x)
        x1 = self.inc2(self.inc1(x))
        x2 = self.conv2(self.down1(x1))
        x3 = self.conv1(self.down2(x2))
        x4 = self.conv0(self.down3(x3))
        x5 = self.conv0(self.down4(x4))
        x = self.dropout(self.conv1(self.up1(x5, x4)))
        x = self.dropout(self.conv2(self.up2(x, x3)))
        x = self.dropout(self.conv3(self.up3(x, x2)))
        x = self.dropout(self.conv3(self.up4(x, x1)))
        x = self.outc(x)
        return x


