import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from numba import jit
import torchvision.models as models

from src.features.my_BEV_Unet import BEV_Unet

class ptBEVnet(nn.Module):
    def __init__(
        self,
        backbone_name,
        grid_size,
        projection_type,
        n_class,
        out_pt_fea_dim=512,
        max_pt_per_encode=256,
        circular_padding=False,
        device=0
    ):
        super(ptBEVnet, self).__init__()


        self.max_pt = max_pt_per_encode
        self.backbone_name = backbone_name
        self.n_height = grid_size[2]
        self.grid_size = grid_size
        self.n_class = len(n_class)
        self.circular_padding = circular_padding
        self.device = device

        if projection_type == "traditional":
            fea_dim = 7
        elif projection_type == "polar":
            fea_dim = 9
        else:
            AssertionError, "incorrect projection type"

        self.Simplified_PointNet = nn.Sequential(
            nn.BatchNorm1d(fea_dim),
            nn.Linear(fea_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, out_pt_fea_dim),
        )

        self.make_backbone_input_fea_dim = nn.Sequential(nn.Linear(out_pt_fea_dim, self.n_height), nn.LeakyReLU())

        assert self.backbone_name in ["UNet", "FCN", "DL"], "backbone name is incorrect"

        if self.backbone_name == "UNet":
            self.backbone = BEV_Unet(self.n_class, self.n_height, self.circular_padding)
            
        elif self.backbone_name == "FCN":
            self.backbone = models.segmentation.fcn_resnet50(pretrained=False, pretrained_backbone=False, num_classes=19*32).to(self.device)
            self.backbone.backbone.conv1 = nn.Conv2d(self.n_height, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(self.device)
        elif self.backbone_name == "DL":
            self.backbone = models.segmentation.deeplabv3_resnet101(pretrained=False, pretrained_backbone=False, num_classes=19*32).to(self.device)
            self.backbone.backbone.conv1 = nn.Conv2d(self.n_height, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(self.device)

    def forward(self, pt_fea, xy_ind, device):
        batch_size = len(pt_fea)
        backbone_input_dim = [batch_size, self.grid_size[0], self.grid_size[1], self.n_height]
        backbone_data = torch.zeros(backbone_input_dim, dtype=torch.float32).to(device)

        fea, ind = [], []
        num = 0
        for i in range(batch_size):
            pt_num = len(pt_fea[i])
            batch_ind = torch.cat([torch.full((pt_num,1), i).to(device), xy_ind[i]], dim=1)
            fea.append(pt_fea[i])
            ind.append(batch_ind)
            num += pt_num
        fea, ind = torch.cat(fea, dim=0).to(device), torch.cat(ind, dim=0).to(device)
            
        random_ind = torch.randperm(num, device=device)
        fea, ind = torch.index_select(fea, dim=0, index=random_ind), torch.index_select(ind, dim=0, index=random_ind)
        unq, unq_inv = torch.unique(ind, return_inverse=True, dim=0)
        ## Own implementation of torch.unique() function TODO: fix speed problem
        # x_sort_ind = batch_xy_ind[batch_xy_ind[:,1].sort()[1]]
        # sort_xy_ind = x_sort_ind[x_sort_ind[:,0].sort()[1]]
        # unq, unq_inv, unq_cnt = index_sort(sort_xy_ind.detach().cpu().numpy(),
        #                                    batch_xy_ind.detach().cpu().numpy(),
        #                                    pt_num)
        # unq = torch.tensor(unq, dtype=torch.int64, device = device)
        # unq_inv = torch.tensor(unq_inv, dtype=torch.int64, device = device)
        # unq_cnt = torch.tensor(unq_cnt, dtype=torch.int64, device = device)

        pointnet_fea = self.Simplified_PointNet(fea)
        max_pointnet_fea = torch_scatter.scatter_max(pointnet_fea, unq_inv, dim=0)[0]

        backbone_input_fea = self.make_backbone_input_fea_dim(max_pointnet_fea)
        backbone_data[unq[:,0],unq[:,1],unq[:,2],:] = backbone_input_fea

        to_backbone = backbone_data.permute(0, 3, 1, 2)
        backbone_fea = self.backbone(to_backbone)

        if self.backbone_name != "UNet":
            backbone_fea = torch.tensor(backbone_fea['out']).to(device)
            class_per_voxel_dim = [batch_size, self.grid_size[0], self.grid_size[1], self.n_height, self.n_class]
            backbone_fea = backbone_fea.permute(0,2,3,1)
            backbone_fea = backbone_fea.view(class_per_voxel_dim).permute(0,4,1,2,3)

        return backbone_fea

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


# TODO: cite or remove
def grp_range_torch(a, dev):
    idx = torch.cumsum(a, 0)
    id_arr = torch.ones(idx[-1], dtype=torch.int64, device=dev)
    id_arr[0] = 0
    id_arr[idx[:-1]] = -a[:-1] + 1
    return torch.cumsum(id_arr, 0)
