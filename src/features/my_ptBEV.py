import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dropblock import DropBlock2D

class ptBEVnet(nn.Module):
    
    def __init__(self, BEV_Unet, pytorch_device, grid_size,  fea_dim = 7, out_pt_fea_dim = 512, max_pt_per_encode = 256, n_height = 32):
        super(ptBEVnet, self).__init__()
        
        self.device = pytorch_device
        self.max_pt = max_pt_per_encode
        self.n_height = n_height
        self.grid_size = grid_size

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
        
        self.BEVUNet = BEV_Unet

        
    def forward(self, pt_fea, xy_ind, circular_padding):
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
 
        return unet_fea

#TODO undertanding grp_range_torch
def grp_range_torch(a,gpu):
    idx = torch.cumsum(a,0)
    id_arr = torch.ones(idx[-1],dtype = torch.int64,device=gpu)
    id_arr[0] = 0
    id_arr[idx[:-1]] = -a[:-1]+1
    return torch.cumsum(id_arr,0)