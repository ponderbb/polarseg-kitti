import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dropblock import DropBlock2D

class BEVnet(nn.Module):
    
    def __init__(self, UNet, pytorch_device, grid_size,  fea_dim = 7, out_pt_fea_dim = 512, max_pt_per_encode = 256, n_height = 32, 
                n_class=19, dropout = 0.5, circular_padding = True, dropblock = True):
        super(BEVnet, self).__init__()
        
        self.device = pytorch_device
        self.max_pt = max_pt_per_encode
        self.n_height = n_height
        self.grid_size = grid_size
        self.circular_padding = circular_padding

        self.Simplified_PointNet = nn.Sequential(
            nn.BatchNorm1d(fea_dim),
            
            nn.Linear(fea_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, out_pt_fea_dim)
        )
        
        self.fea_compression = nn.Sequential(
            nn.Linear(out_pt_fea_dim, self.n_height),
            nn.ReLU())
        
        self.BEVUNet = UNet(n_class*n_height, n_height, dropout, circular_padding, dropblock)

        
    def forward(self, pt_fea, xy_ind, circular_padding= True):
        batch_size = len(pt_fea)
        batch_out_data_dim = [self.grid_size[0], self.grid_size[1], self.n_height]
        out = []
        for i in range(batch_size):
            batch_pt_fea = pt_fea[i]
            batch_xy_ind = xy_ind[i]
            pt_num = len(pt_fea[i])

            shuffled_ind = torch.randperm(pt_num, device = self.device)
            batch_pt_fea = torch.index_select(batch_pt_fea, dim=0, index=shuffled_ind)
            batch_xy_ind = torch.index_select(batch_xy_ind, dim=0, index=shuffled_ind)

            unq, unq_inv, unq_cnt = torch.unique(batch_xy_ind,return_inverse=True, return_counts=True, dim=0)
            unq = unq.type(torch.int64)
        
            grp_ind = grp_range_torch(unq_cnt, self.device)[torch.argsort(torch.argsort(unq_inv))]
            remain_ind = grp_ind < self.max_pt
            
            batch_pt_fea = batch_pt_fea[remain_ind,:]
            unq_inv = unq_inv[remain_ind]
        
            pointnet_fea = self.Simplified_PointNet(batch_pt_fea)
            max_pointnet_fea_list = []
            for i in range(len(unq_cnt)):
                max_pointnet_fea_list.append(torch.max(pointnet_fea[unq_inv==i],dim=0)[0])
            max_pointnet_fea = torch.stack(max_pointnet_fea_list)
            nheight_pointnet_fea = self.fea_compression(max_pointnet_fea)

            batch_out_data = torch.zeros(batch_out_data_dim, dtype=torch.float32).to(self.device)
            batch_out_data[unq[:,0],unq[:,1],:] = nheight_pointnet_fea
            out.append(batch_out_data)
            
        to_cnn = torch.stack(out)
        to_cnn = to_cnn.permute(0,3,1,2)

        unet_fea = self.BEVUNet(to_cnn, circular_padding)
        cnn_result = unet_fea.permute(0,2,3,1)
        new_shape = list(cnn_result.size())[:3] + [self.n_height,self.n_class]
        cnn_result = cnn_result.view(new_shape)
        cnn_result = cnn_result.permute(0,4,1,2,3)
        return cnn_result

#TODO undertanding grp_range_torch
def grp_range_torch(a,gpu):
    idx = torch.cumsum(a,0)
    id_arr = torch.ones(idx[-1],dtype = torch.int64,device=gpu)
    id_arr[0] = 0
    id_arr[idx[:-1]] = -a[:-1]+1
    return torch.cumsum(id_arr,0)


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
        self.pooling = nn.MaxPool2d(2),
        self.conv = double_CBR(in_ch, out_ch, circular_padding)
      
    def forward(self, x, circular_padding):
        x = self.pooling(x)
        x = self.conv(x, circular_padding)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, circular_padding, bilinear=True, use_dropblock = False, drop_p = 0.5):
        super(up, self).__init__()
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
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