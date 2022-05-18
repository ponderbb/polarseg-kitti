import numpy as np
import torch
import yaml
from numba import jit
from torch.utils.data import Dataset

import src.misc.utils as utils


class SemanticKITTI(Dataset):
    def __init__(self, path, data_split="train", reflection=True, fixed_volume=True) -> None:

        with open("semantic-kitti.yaml", "r") as stream:
            dataset_yaml = yaml.safe_load(stream)

        self.data_split = data_split
        self.reflection = reflection
        self.fixed_volume = fixed_volume
        self.learning_map = dataset_yaml["learning_map"]
        self.scan_list = []
        self.label_list = []
        try:
            split = dataset_yaml["split"][self.data_split]
        except ValueError:
            print("Incorrect set type")

        for sequence_folder in split:
            self.scan_list += utils.getPath("/".join([path, "sequences", str(sequence_folder).zfill(2), "velodyne"]))
            self.label_list += utils.getPath("/".join([path, "sequences", str(sequence_folder).zfill(2), "labels"]))
        self.scan_list.sort()
        self.label_list.sort()

    def __len__(self):
        return len(self.scan_list)

    def __getitem__(self, index):
        # scan is containing the (x,y,z, reflection)
        scan = np.fromfile(self.scan_list[index], dtype=np.float32).reshape(-1, 4)
        if self.data_split == "test":
            labels = np.zeros(shape=scan[:, 0].shape, dtype=int)
        else:
            labels = np.fromfile(self.label_list[index], dtype=np.int32)
            labels = labels & 0xFFFF  # according to the semanticKITTI apilab
            labels[list(self.learning_map.keys())] = list(self.learning_map.values())  # remap from cross-entropy labels
            labels = labels.reshape(-1, 1)

        if self.reflection:
            data_tuple = (scan, labels)
        else:
            data_tuple = (scan[:, :3], labels)

        return data_tuple


class cart_voxel_dataset(Dataset):
    def __init__(
        self,
        in_dataset,
        grid_size,
        fixed_volume=False,
        max_volume: list = [50, 50, 1.5],
        min_volume: list = [-50, -50, -3],
        flip_augmentation=False,
        random_rotation=False,
    ):
        self.in_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.fixed_vol = fixed_volume
        self.max_vol = np.asarray(max_volume, dtype=np.float32)
        self.min_vol = np.asarray(min_volume, dtype=np.float32)
        self.flip_aug = flip_augmentation
        self.rot_aug = random_rotation

    def __len__(self):
        return len(self.in_dataset)

    def __getitem__(self, index):

        # extrat data
        data, labels = self.in_dataset[index]
        xyz = data[:, :3]
        reflection = data[:, 3]

        # TODO: augmentations

        # fix volume space
        ROI = self.max_vol - self.min_vol
        intervals = ROI / (self.grid_size - 1)

        # calculate the grid indices
        if self.fixed_vol:
            xyz = utils.clip(xyz, self.min_vol, self.max_vol)

        # calculate the grid index for each point
        grid_index = np.floor(xyz - self.min_vol / intervals).astype(np.int)  # NOTE: cite this

        # get the coordinates of the voxels
        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        voxel_position = np.indices(self.grid_size) * intervals.reshape([-1, 1, 1, 1]) + self.min_vol.reshape(
            [-1, 1, 1, 1]
        )

        # process the labels and vote for one per voxel

        voxel_label = np.zeros(self.grid_size, dtype=np.uint8)
        raw_point_label = np.concatenate([grid_index, labels.reshape(-1, 1)], axis=1)
        sorted_point_label = raw_point_label[
            np.lexsort((grid_index[:, 0], grid_index[:, 1], grid_index[:, 2])), :
        ].astype(np.int64)
        voxel_label = label_voting(np.copy(voxel_label), sorted_point_label)

        # center points on voxel
        voxel_center = (grid_index.astype(float) + 0.5) * intervals + self.min_vol
        centered_xyz = xyz - voxel_center
        pt_features = np.concatenate((centered_xyz, xyz, reflection.reshape(-1, 1)), axis=1)

        # TODO: version data_tuple based on arguments
        data_tuple = (voxel_label, grid_index, labels, pt_features)
        """
        *data_tuple*
        ---
        voxel_label: voxel-level label
        grid_index: individual point's grid index
        labels: individual point's label
        pt_features: [centered xyz, xyz, reflection]
        """
        return data_tuple


@jit("u1[:,:,:](u1[:,:,:],i8[:,:])", nopython=True, cache=True, parallel=False)
def label_voting(voxel_label: np.array, sorted_list: list):

    # FIXME: way to similar to the original, figure out how to do it differenlty (only do a lookup array for existing labels or sth)
    # TODO: add numba decorator to speed up process
    label_counter = np.zeros((256,), dtype=np.uint)
    label_counter[sorted_list[0, 3]] = 1
    compare_label_a = sorted_list[0, :3]
    for i in range(1, sorted_list.shape[0]):
        compare_label_b = sorted_list[i, :3]
        if not np.all(compare_label_a == compare_label_b):
            voxel_label[compare_label_a[0], compare_label_a[1], compare_label_a[2]] = np.argmax(label_counter)
            compare_label_a = compare_label_b
            label_counter = np.zeros((256,), dtype=np.uint)
        label_counter[sorted_list[i, 3]] += 1
    voxel_label[compare_label_a[0], compare_label_a[1], compare_label_a[2]] = np.argmax(label_counter)
    return voxel_label


def main():

    semkitti = SemanticKITTI(path="/root/repos/polarseg-kitti/data/debug", data_split="train")
    train_dataset = cart_voxel_dataset(semkitti, grid_size=[480, 360, 32], fixed_volume=True)
    dummy_dataloader = torch.utils.data.DataLoader(train_dataset)

    for data_tuple in dummy_dataloader:
        print("__")


if __name__ == "__main__":
    main()
