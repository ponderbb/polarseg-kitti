import torch.nn as nn
import torch
import torch.nn.functional as F

from src.features.layer_functions import CBR, BlockDrop, down_CBR, up_CBR


class Unet(nn.Module):
    def __init__(self, n_class, n_height, circular_padding=False):
        super(Unet, self).__init__()
        self.n_class = n_class
        self.n_height = n_height
        self.circular_padding = circular_padding

        self.inc = nn.Sequential(
            nn.BatchNorm2d(n_height),
            CBR(n_height, 64, self.circular_padding), 
            CBR(64, 64, self.circular_padding)
        )

        self.down1 = nn.Sequential(
            down_CBR(64, 128, self.circular_padding),
            CBR(128, 128, self.circular_padding)
        )

        self.down2 = nn.Sequential(
            down_CBR(128, 256, self.circular_padding),
            CBR(256, 256, self.circular_padding)
        )

        self.down3 = nn.Sequential(
            down_CBR(256, 512, self.circular_padding),
            CBR(512, 512, self.circular_padding)
        )

        self.down4 = nn.Sequential(
            down_CBR(512, 512, self.circular_padding),
            CBR(512, 512, self.circular_padding)
        )


        self.up1 = up_CBR(1024, 256, self.circular_padding)
        self.up1_sub = nn.Sequential(
            CBR(256, 256, self.circular_padding),
            BlockDrop(drop_p=0.5, block_size=7)
        )
        self.up2 = up_CBR(512, 128, self.circular_padding)
        self.up2_sub = nn.Sequential(
            CBR(128, 128, self.circular_padding),
            BlockDrop(drop_p=0.5, block_size=7)
        )
        self.up3 = up_CBR(256, 64, self.circular_padding)
        self.up3_sub = nn.Sequential(
            CBR(64, 64, self.circular_padding),
            BlockDrop(drop_p=0.5, block_size=7)
        )
        self.up4 = up_CBR(128, 64, self.circular_padding)        
        self.up4_sub = nn.Sequential(
            CBR(64, 64, self.circular_padding),
            BlockDrop(drop_p=0.5, block_size=7)
        )

        self.outc = nn.Conv2d(64, n_class * n_height, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1_sub(self.up1(x5, x4))
        x = self.up2_sub(self.up2(x, x3))
        x = self.up3_sub(self.up3(x, x2))
        x = self.up4_sub(self.up4(x, x1))
        x = self.outc(x)
        x = x.permute(0, 2, 3, 1)
        class_per_voxel_dim = [x.size()[0], x.size()[1], x.size()[2], self.n_height, self.n_class]
        x = x.view(class_per_voxel_dim).permute(0, 4, 1, 2, 3)
        return x
