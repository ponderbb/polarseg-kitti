import os
import numpy as np
import torch
import random
import time
import numba as nb
import yaml
from torch.utils import data


class spherical_projection(data.Dataset):
    def __init__(self, in_dataset, grid_size = [64,2048,2], fov_up=3.0, fov_down=-25.0, ignore_label = 0,return_test = False,):
        self.point_cloud_dataset = in_dataset
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.proj_H = grid_size[0]
        self.proj_W = grid_size[1]
        self.proj_D = grid_size[2]
        self.grid_size = grid_size
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz,labels = data
        elif len(data) == 3:
            xyz,labels,sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else: raise Exception('Return invalid data tuple')

        point_num = len(xyz)
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi 
        fov_down = self.proj_fov_down / 180.0 * np.pi 
        fov = abs(fov_down) + abs(fov_up) 

        depth = np.linalg.norm(xyz, 2, axis=1)
        max_depth = np.floor(np.max(depth))
        min_depth = np.floor(np.min(depth))
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        
        yaw = -np.arctan2(y, x)
        pitch = np.arcsin(z / depth)

        proj_w = (( 0.5 * (yaw / np.pi + 1.0) ) * self.proj_W )
        proj_h = (( 1.0 - (pitch + abs(fov_down)) / fov ) * self.proj_H )
        depth = depth.reshape(point_num,1)

        proj_x_ind = np.floor(proj_h)
        proj_x_ind = np.minimum(self.proj_H - 1, proj_x_ind)
        proj_x_ind = np.maximum(0, proj_x_ind).astype(np.int32).reshape(point_num,1)

        proj_y_ind = np.floor(proj_w)
        proj_y_ind= np.minimum(self.proj_W - 1, proj_y_ind)
        proj_y_ind = np.maximum(0, proj_y_ind).astype(np.int32).reshape(point_num,1)

        crop_range = max_depth - min_depth        
        intervals = crop_range/(self.proj_D-1)
        proj_x_ind = (np.floor((np.clip(depth,min_depth,max_depth)-min_depth)/intervals)).astype(np.int).reshape(point_num, 1)


        '''
        order = np.argsort(depth[:,0])[::-1]
        indices = np.arange(depth.shape[0])

        order_xyz = xyz[order]
        order_proj_x_ind = proj_x_ind[order]
        order_proj_y_ind = proj_y_ind[order]
        order_labels = labels[order]
        
        range_xyz = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32)
        range_xyz[order_proj_x_ind, order_proj_y_ind] = order_xyz
        

        range_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)
        range_idx[order_proj_x_ind, order_proj_y_ind] = indices

        proj_mask = np.zeros((self.proj_H, self.proj_W),dtype=np.int32)
        proj_mask = (range_idx > 0).astype(np.int32)
        '''
        
        grid_xy_ind = np.concatenate(([proj_x_ind, proj_y_ind]), axis=1)
        grid_z_ind = np.zeros(shape=(point_num,1))
        grid_z_ind[depth>(max_depth-min_depth)/2] = 1
        grid_ind = np.concatenate(([grid_xy_ind, grid_z_ind]), axis=1).astype(np.int)


        voxel_position = np.zeros(self.grid_size,dtype = np.float32)
        dim_array = np.ones(len(self.grid_size)+1,int)
        dim_array[0] = -1 
        voxel_position = np.indices(self.grid_size).reshape(dim_array)

        processed_label = np.ones(self.grid_size,dtype = np.uint8)*self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind,labels],axis = 1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:,0],grid_ind[:,1],grid_ind[:,2])),:]
        processed_label = nb_process_label(np.copy(processed_label),label_voxel_pair)
        
        data_tuple = (voxel_position , processed_label)

        proj_xy = np.concatenate((proj_h.reshape(point_num,1),proj_w.reshape(point_num,1)),axis=1)
        proj_xyz = np.concatenate((proj_xy, depth), axis=1)
        return_xyz = np.concatenate((proj_xyz, xyz),axis=1)
        
        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz,sig[...,np.newaxis]),axis = 1)
        
        if self.return_test:
            data_tuple += (grid_ind,labels,return_fea,index)
        else:
            data_tuple += (grid_ind,labels,return_fea)
        return data_tuple

@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])',nopython=True,cache=True,parallel = False)
def nb_process_label(processed_label,sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,),dtype = np.uint16)
    counter[sorted_label_voxel_pair[0,3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0,:3]
    for i in range(1,sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i,:3]
        if not np.all(np.equal(cur_ind,cur_sear_ind)):
            processed_label[cur_sear_ind[0],cur_sear_ind[1],cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,),dtype = np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i,3]] += 1
    processed_label[cur_sear_ind[0],cur_sear_ind[1],cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


# load Semantic KITTI class info
with open("semantic-kitti.yaml", 'r') as stream:
    semkittiyaml = yaml.safe_load(stream)
SemKITTI_label_name = dict()
for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
    SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]