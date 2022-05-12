import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numba as nb
import torch_scatter

class ptBEVnet(nn.Module):
    
    def __init__(self, pytorch_device, BEV_net, grid_size,  fea_dim = 7, out_pt_fea_dim = 512, max_pt_per_encode = 256, fea_compre = 32):
        super(ptBEVnet, self).__init__()
        
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
        
        self.device = pytorch_device
        self.BEV_model = BEV_net
        self.max_pt = max_pt_per_encode
        self.fea_compre = fea_compre
        self.grid_size = grid_size
        
        self.fea_compression = nn.Sequential(
            nn.Linear(out_pt_fea_dim, self.fea_compre),
            nn.ReLU())
        
        
    def forward(self, pt_fea, xy_ind):
        batch_size = len(pt_fea)
        batch_out_data_dim = [self.grid_size[0], self.grid_size[1], self.fea_compre]
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
            batch_xy_ind = batch_xy_ind[remain_ind,:]
            unq_inv = unq_inv[remain_ind]
        
            pointnet_fea = self.Simplified_PointNet(batch_pt_fea)
            max_pointnet_fea = torch_scatter.scatter_max(pointnet_fea, unq_inv, dim=0)[0]
            nheight_pointnet_fea = self.fea_compression(max_pointnet_fea)

            batch_out_data = torch.zeros(batch_out_data_dim, dtype=torch.float32).to(self.device)
            batch_out_data[unq[:,0],unq[:,1],:] = nheight_pointnet_fea
            out.append(batch_out_data)
            
        to_cnn = torch.stack(out)
        to_cnn = to_cnn.permute(0,3,1,2)
        cnn_result = self.BEV_model(to_cnn)
        
        return cnn_result
    
def grp_range_torch(a,gpu):
    idx = torch.cumsum(a,0)
    id_arr = torch.ones(idx[-1],dtype = torch.int64,device=gpu)
    id_arr[0] = 0
    id_arr[idx[:-1]] = -a[:-1]+1
    return torch.cumsum(id_arr,0)