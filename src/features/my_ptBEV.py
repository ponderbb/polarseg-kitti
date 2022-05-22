import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import jit


class ptBEVnet(nn.Module):
    def __init__(self, BEV_Unet, grid_size, fea_dim=7, out_pt_fea_dim=512, max_pt_per_encode=256, n_height=32):
        super(ptBEVnet, self).__init__()

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
            nn.Linear(256, out_pt_fea_dim),
        )

        self.fea_compression = nn.Sequential(nn.Linear(out_pt_fea_dim, self.n_height), nn.ReLU())

        self.BEVUNet = BEV_Unet

    def forward(self, pt_fea, xy_ind, circular_padding, device):
        batch_size = len(pt_fea)
        batch_out_data_dim = [self.grid_size[0], self.grid_size[1], self.n_height]
        out = []
        for i in range(batch_size):
            batch_pt_fea = pt_fea[i]
            batch_xy_ind = xy_ind[i]
            pt_num = len(pt_fea[i])

            shuffled_ind = torch.randperm(pt_num, device=device)
            batch_pt_fea = torch.index_select(batch_pt_fea, dim=0, index=shuffled_ind)
            batch_xy_ind = torch.index_select(batch_xy_ind, dim=0, index=shuffled_ind)

            # unq, unq_inv, unq_cnt= torch.unique(batch_xy_ind, return_inverse=True,return_counts=True, dim=0)
            x_sort_ind = batch_xy_ind[batch_xy_ind[:, 1].sort()[1]]
            sort_xy_ind = x_sort_ind[x_sort_ind[:, 0].sort()[1]]
            unq, unq_inv, unq_cnt = index_sort(
                sort_xy_ind.detach().cpu().numpy(), batch_xy_ind.detach().cpu().numpy(), pt_num
            )
            unq = torch.tensor(unq, dtype=torch.int64, device=device)
            unq_inv = torch.tensor(unq_inv, dtype=torch.int64, device=device)
            unq_cnt = torch.tensor(unq_cnt, dtype=torch.int64, device=device)

            grp_ind = grp_range_torch(unq_cnt, device)[torch.argsort(torch.argsort(unq_inv))]
            remain_ind = grp_ind < self.max_pt

            batch_pt_fea = batch_pt_fea[remain_ind, :]
            batch_xy_ind = batch_xy_ind[remain_ind, :]
            unq_inv = unq_inv[remain_ind]

            pointnet_fea = self.Simplified_PointNet(batch_pt_fea)

            # max_pointnet_fea = torch_scatter.scatter_max(pointnet_fea, unq_inv, dim=0)[0]
            unq_num = len(unq_cnt)
            max_pointnet_fea = get_max_fea(unq_num, pointnet_fea, unq_inv)

            nheight_pointnet_fea = self.fea_compression(max_pointnet_fea)

            batch_out_data = torch.zeros(batch_out_data_dim, dtype=torch.float32).to(device)
            batch_out_data[unq[:, 0], unq[:, 1], :] = nheight_pointnet_fea
            out.append(batch_out_data)

        to_cnn = torch.stack(out)
        to_cnn = to_cnn.permute(0, 3, 1, 2)

        unet_fea = self.BEVUNet(to_cnn, circular_padding)

        return unet_fea


@jit(nopython=True)
def index_sort(sort_ind, xy_ind, pt_num):
    unq = [sort_ind[0]]
    unq_cnt = []
    count = 0
    for ind in sort_ind:
        if ind[0] == unq[-1][0] and ind[1] == unq[-1][1]:
            count += 1
        else:
            unq.append(ind)
            unq_cnt.append(count)
            count = 1
    unq_cnt.append(count)

    unq_inv = [0 for i in range(pt_num)]

    for i, ind in enumerate(unq):
        boolen = np.where(xy_ind == ind, True, False)
        condition = []
        for j, boo in enumerate(boolen):
            tf = np.all(boo)
            if tf:
                condition.append(j)
        for cond in condition:
            unq_inv[cond] = i
    return unq, unq_inv, unq_cnt


def get_max_fea(num, feature, unq_inv):
    max_pointnet_fea_list = []
    for i in range(num):
        condition = unq_inv == i
        max_pointnet_fea_list.append(torch.max(feature[condition], dim=0)[0])
    max_pointnet_fea = torch.stack(max_pointnet_fea_list)
    return max_pointnet_fea


def grp_range_torch(a, device):
    idx = torch.cumsum(a, 0)
    id_arr = torch.ones(idx[-1], dtype=torch.int64, device=device)
    id_arr[0] = 0
    id_arr[idx[:-1]] = -a[:-1] + 1
    return torch.cumsum(id_arr, 0)
