import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D

class Res_block(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, circular_padding):
        super(Res_block,self).__init__()
        self.relu = nn.ReLU()
        self.circular_padding = circular_padding
        if in_ch == 64 or in_ch==out_ch:
            if self.circular_padding :
                self.convseq = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 1, padding=0),
                nn.BatchNorm2d(mid_ch),
                nn.LeakyReLU(inplace=True),
                polar_CBR(mid_ch, mid_ch, False),
                nn.Conv2d(mid_ch, out_ch, 1, padding=0),
                nn.BatchNorm2d(out_ch),
                )

            else : 
                self.convseq = nn.Sequential(
                    nn.Conv2d(in_ch, mid_ch, 1, padding=0),
                    nn.BatchNorm2d(mid_ch),
                    nn.LeakyReLU(inplace=True),
                    CBR(mid_ch, mid_ch, False),
                    nn.Conv2d(mid_ch, out_ch, 1, padding=0),
                    nn.BatchNorm2d(out_ch),
                )
            if in_ch == 64: 
                self.iden = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)
            else : 
                self.iden = nn.Identity()
        
        else:
            if self.circular_padding:
                self.convseq = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 1,2, 0),
                nn.BatchNorm2d(mid_ch),
                nn.LeakyReLU(inplace=True),
                polar_CBR(mid_ch, mid_ch, False),
                nn.Conv2d(mid_ch, out_ch, 1, padding=0),
                nn.BatchNorm2d(out_ch),
                )
            else :
                self.convseq = nn.Sequential(
                    nn.Conv2d(in_ch, mid_ch, 1,2, 0),
                    nn.BatchNorm2d(mid_ch),
                    nn.LeakyReLU(inplace=True),
                    CBR(mid_ch, mid_ch, False),
                    nn.Conv2d(mid_ch, out_ch, 1, padding=0),
                    nn.BatchNorm2d(out_ch),
                )
            self.iden = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2)
        
    def forward(self, x):
        y = self.convseq(x) 
        x = y + self.iden(x)
        x = self.relu(x)
        return x
        
class polar_CBR(nn.Module):
    def __init__(self, in_ch, out_ch, double_version=False, kernel_size = 3, stride = 1, padding = 1):
        super(polar_CBR, self).__init__()
        self.double_version = double_version
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cbr1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, self.kernel_size, self.stride, (self.padding,0)),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
        self.cbr2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, self.kernel_size, self.stride, (self.padding,0)),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = F.pad(x,(self.padding,self.padding,0,0),mode = 'circular')
        x = self.cbr1(x)
        if self.double_version :
            x = F.pad(x,(self.padding,self.padding,0,0),mode = 'circular')
            x = self.cbr2(x)
        return x

class CBR(nn.Module):
    def __init__(self, in_ch, out_ch, double_version=False, kernel_size = 3, stride = 1, padding = 1):
        super(CBR, self).__init__()
        self.double_version = double_version
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.cbr1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
        self.cbr2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.cbr1(x)
        if self.double_version:
            x = self.cbr2(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, circular_padding, double_version=False):
        super(inconv, self).__init__()
        self.double_version = double_version

        if circular_padding:
            self.cbr = nn.Sequential(
                nn.BatchNorm2d(in_ch),
                polar_CBR(in_ch, out_ch, self.double_version)
            )

        else:
            self.cbr = nn.Sequential(
                nn.BatchNorm2d(in_ch),
                CBR(in_ch, out_ch, self.double_version)
            )
        
    def forward(self, x):
        x = self.cbr(x)
        return x

class up_CBR(nn.Module):
    def __init__(self, in_ch, out_ch, circular_padding, mode="UNet", double_version=False):
        super(up_CBR, self).__init__()
        self.circular_padding = circular_padding
        self.mode = mode
        self.double_version = double_version
        self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2, groups=in_ch//2)
        self.drop  = DropBlock2D(0.5, 7)

        if mode == "UNet":
            if circular_padding :
                self.cbr = polar_CBR(in_ch, out_ch, self.double_version)
            else :
                self.cbr = CBR(in_ch, out_ch, self.double_version)

        if mode == "FCN":
            if circular_padding :
                self.cbr = polar_CBR(in_ch//2 + out_ch, out_ch, self.double_version)
            else:
                self.cbr = CBR(in_ch//2 + out_ch, out_ch, self.double_version)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        up_error_x, up_error_y = x2.size()[3] - x1.size()[3], x2.size()[2] - x1.size()[2]
        left_padding, right_padding = up_error_x//2, up_error_x - up_error_x//2
        up_padding, down_padding = up_error_y//2, up_error_y - up_error_y//2
        x1 = F.pad(x1, (left_padding, right_padding, up_padding, down_padding))

        x = torch.cat([x2, x1], dim=1)
        x = self.cbr(x)

        if self.double_version:
            x = self.drop(x)
        return x


class down_CBR(nn.Module):
    def __init__(self, in_ch, out_ch, circular_padding, double_version):
        super(down_CBR, self).__init__()
        self.circular_padding = circular_padding
        self.double_version = double_version
        self.pooling = nn.MaxPool2d(2)
        if circular_padding :
            self.conv = polar_CBR(in_ch, out_ch, self.double_version)
        else:
            self.conv = CBR(in_ch, out_ch,self.double_version)

    def forward(self, x):
        x = self.pooling(x)
        x = self.conv(x)
        return x

def reshape_to_voxel(x, n_height, n_class):
    x = x.permute(0,2,3,1)
    voxel_shape =  [x.size()[0], x.size()[1], x.size()[2], n_height, n_class]
    x = x.view(voxel_shape).permute(0,4,1,2,3)
    return x

