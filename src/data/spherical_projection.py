import numpy as np
from torch.utils import data
import numba as nb

#TODO we have differnce view point !!! 

class spherical_projection(data.Dataset):
    def __init__(self, in_dataset, gird_size = [480, 2048, 28], H=64, W=2048, depths = 480, fov_up=3.0, fov_down=-25.0, ignore_label = 255,return_test = False,):
        self.point_cloud_dataset = in_dataset
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.proj_H = H
        self.proj_W = W
        self.proj_D = depths
        self.grid_size = gird_size
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

        depth = np.linalg.norm(xyz, 2, axis=1).reshape(point_num, 1)  
        max_depth = np.floor(np.max(depth))
        min_depth = 0

        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        
        yaw = -np.arctan2(y, x)
        pitch = np.arcsin(z / depth)

        proj_w = (( 0.5 * (yaw / np.pi + 1.0) ) * self.proj_W ).reshape(point_num, 1)  
        proj_h = (( 1.0 - (pitch + abs(fov_down)) / fov ) * self.proj_H ).reshape(point_num, 1)  
        
        proj_xy = np.concatenate([depth, proj_w], axis=1)
        proj_xyz = np.concatenate([proj_xy, proj_h], axis=1)

        proj_y_ind = np.floor(proj_w)
        proj_y_ind= np.minimum(self.proj_W - 1, proj_y_ind)
        proj_y_ind = np.maximum(0, proj_y_ind).astype(np.int32).reshape(point_num, 1)  

        proj_z_ind = np.floor(proj_h)
        proj_z_ind = np.minimum(self.proj_H - 1, proj_z_ind)
        proj_z_ind = np.maximum(0, proj_z_ind).astype(np.int32).reshape(point_num, 1)  

        crop_range = max_depth - min_depth        
        intervals = crop_range/(self.proj_D-1)
        proj_x_ind = (np.floor((np.clip(depth,min_depth,max_depth)-min_depth)/intervals)).astype(np.int).reshape(point_num, 1)
        
        grid_yz_ind = np.concatenate([proj_y_ind, proj_z_ind], axis=1)
        grid_ind = np.concatenate([proj_x_ind, grid_yz_ind], axis=1)

        processed_label = np.ones(self.grid_size, dtype = np.uint8)*self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind,labels],axis = 1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:,0],grid_ind[:,1],grid_ind[:,2])),:]
        processed_label = nb_process_label(np.copy(processed_label),label_voxel_pair)

        data_tuple = (proj_xyz, processed_label)
        
        voxel_x_centers = ((proj_x_ind.astype(np.float32) + 0.5)*intervals + min_depth)
        voxel_yz_centers = (grid_yz_ind.astype(np.float32) + 0.5)
        voxel_centers = np.concatenate([voxel_x_centers, voxel_yz_centers], axis=1)
        return_xyz = proj_xyz - voxel_centers
        return_xyz = np.concatenate((return_xyz,xyz),axis = 1)
        
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