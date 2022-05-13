import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BEV_Unet(nn.Module):

    def __init__(self,n_class,n_height,dropout = 0.5,circular_padding = False):
        super(BEV_Unet, self).__init__()
        self.n_class = n_class
        self.n_height = n_height
        self.network = UNet(n_class*n_height,n_height,dropout,circular_padding)

    def forward(self, x, circular_padding):
        x = self.network(x, circular_padding)
        x = x.permute(0,2,3,1)
        new_shape = list(x.size())[:3] + [self.n_height, self.n_class]
        x = x.view(new_shape)
        x = x.permute(0,4,1,2,3)
        return x


class UNet(nn.Module):

    def __init__(self, n_class,n_height, dropout,circular_padding):
        super(UNet, self).__init__()
        self.norm = nn.BatchNorm2d(n_height)
        self.inc = double_CBR(n_height, 64, circular_padding)
        self.down1 = down(64, 128, circular_padding)
        self.down2 = down(128, 256, circular_padding)
        self.down3 = down(256, 512, circular_padding)
        self.down4 = down(512, 512, circular_padding)
        self.up1 = up(1024, 256, circular_padding, drop_p=dropout)
        self.up2 = up(512, 128, circular_padding, drop_p=dropout)
        self.up3 = up(256, 64, circular_padding, drop_p=dropout)
        self.up4 = up(128, 64, circular_padding, drop_p=dropout)
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
    def __init__(self, in_ch, out_ch, circular_padding, drop_p = 0.5):
        super(up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = double_CBR(in_ch, out_ch, circular_padding)
        self.drop_prob = drop_p

    def forward(self, x1, x2, circular_padding):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x, circular_padding)

        # get gamma value
        gamma = compute_gamma(self.drop_prob, 7)

        # sample mask
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

        # place mask on input device
        mask = mask.to(x.device)

        # compute block mask
        block_mask = compute_block_mask(mask, 7)

        # apply block mask
        out = x * block_mask[:, None, :, :]

        # scale output
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

def compute_gamma(drop_prob, block_size):
    return drop_prob / (block_size ** 2)