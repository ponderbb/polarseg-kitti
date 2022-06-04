import torch.nn as nn
import torch
import torch.nn.functional as F

from src.features.layer_functions import inconv, down_CBR, up_CBR, reshape_to_voxel

class Unet(nn.Module):

    def __init__(self,n_class,n_height,circular_padding = False):
        super(Unet, self).__init__()
        self.n_class = n_class
        self.n_height = n_height
        self.inc = inconv(n_height, 64, circular_padding, double_version = True)
        self.down1 = down_CBR(64, 128, circular_padding, double_version=True)
        self.down2 = down_CBR(128, 256, circular_padding, double_version=True)
        self.down3 = down_CBR(256, 512, circular_padding, double_version=True)
        self.down4 = down_CBR(512, 512, circular_padding, double_version=True)
        self.up1 = up_CBR(1024, 256, circular_padding, double_version=True)
        self.up2 = up_CBR(512, 128, circular_padding, double_version=True)
        self.up3 = up_CBR(256, 64, circular_padding, double_version=True)
        self.up4 = up_CBR(128, 64, circular_padding,double_version= True)
        self.outc = nn.Conv2d(64, n_class*n_height,1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)  
        x = reshape_to_voxel(x, self.n_height, self.n_class)
        return x
