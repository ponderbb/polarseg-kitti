from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from numba import jit
from torch.utils.data import DataLoader, Dataset

import src.misc.utils as utils


class PolarNetDataModule(pl.LightningDataModule):
    def __init__(self, config_path: str = "config/debug.yaml"):
        super().__init__()
        if Path(config_path).exists():
            self.config = utils.load_yaml(config_path)
        else:
            raise FileNotFoundError("Config file can not be found.")
        if Path(self.config["data_dir"]).exists():
            self.data_dir = self.config["data_dir"]
        else:
            raise FileNotFoundError("Data folder can not be found.")
        self.model_type = self.config["model_type"]

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.semkitti_train = SemanticKITTI(self.data_dir, data_split="train")
            self.semkitti_valid = SemanticKITTI(self.data_dir, data_split="valid")
        if stage == "test" or stage is None:
            self.semkitti_test = SemanticKITTI(self.data_dir, data_split="test")

        if self.model_type == "traditional":
            if stage == "fit" or stage is None:
                self.voxelised_train = cart_voxel_dataset(self.config, self.semkitti_train, data_split="train")
                self.voxelised_valid = cart_voxel_dataset(self.config, self.semkitti_valid, data_split="valid")
            if stage == "test" or stage is None:
                self.voxelised_test = cart_voxel_dataset(self.config, self.semkitti_test, data_split="test")
        elif self.model_type == "polar":
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.voxelised_train,
            collate_fn=collate_fn_BEV,
            batch_size=self.config["train_batch"],
            num_workers=self.config["num_workers"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.voxelised_valid,
            collate_fn=collate_fn_BEV,
            batch_size=self.config["valid_batch"],
            num_workers=self.config["num_workers"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.voxelised_test,
            collate_fn=collate_fn_BEV,
            batch_size=self.config["valid_batch"],
            num_workers=self.config["num_workers"],
        )


class SemanticKITTI(Dataset):
    def __init__(self, data_dir: str, data_split) -> None:
        self.data_dir = data_dir
        self.semkitti_yaml = utils.load_yaml("semantic-kitti.yaml")
        self.data_split = data_split
        self.scan_list = []
        self.label_list = []
        try:
            split = self.semkitti_yaml["split"][self.data_split]
        except ValueError:
            print("Incorrect set type")

        for sequence_folder in split:
            self.scan_list += utils.getPath(
                "/".join([self.data_dir, "sequences", str(sequence_folder).zfill(2), "velodyne"])
            )
            self.label_list += utils.getPath(
                "/".join([self.data_dir, "sequences", str(sequence_folder).zfill(2), "labels"])
            )
        self.scan_list.sort()
        self.label_list.sort()

    def __len__(self):
        return len(self.scan_list)

    def __getitem__(self, index):
        # scan is containing the (x,y,z, reflection)
        scan = np.fromfile(self.scan_list[index], dtype=np.float32).reshape(-1, 4)

        # labels prepared based on semkitti documentation
        if self.data_split == "test":
            labels = np.zeros(shape=scan[:, 0].shape, dtype=int)
        else:
            labels = np.fromfile(self.label_list[index], dtype=np.int32).reshape(-1, 1)
            labels = labels & 0xFFFF  # cut upper half of the binary (source: semkittiAPI)
            labels = utils.remap_labels(labels, self.semkitti_yaml).reshape(
                -1, 1
            )  # remap to cross-entropy labels (source: semkittiAPI)

        return (scan, labels)


class cart_voxel_dataset(Dataset, PolarNetDataModule):
    def __init__(
        self,
        config: dict,
        dataset,
        data_split: str,
    ):
        self.dataset = dataset
        self.unlabeled_idx = utils.ignore_class(config["semkitti_config"])
        self.grid_size = np.asarray(config["grid_size"])
        self.max_vol = np.asarray(config["max_vol"], dtype=np.float32)
        self.min_vol = np.asarray(config["min_vol"], dtype=np.float32)
        self.fixed_vol = config["augmentations"]["fixed_vol"]
        self.flip_aug = config["augmentations"]["flip"]
        self.rot_aug = config["augmentations"]["rot"]
        self.reflection = config["reflection"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        # extrat data
        data, labels = self.dataset[index]
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
        grid_index = np.floor(xyz - self.min_vol / intervals).astype(int)  # NOTE: cite this

        # process the labels and vote for one per voxel
        voxel_label = np.full(self.grid_size, self.unlabeled_idx, dtype=np.uint8)
        raw_point_label = np.concatenate([grid_index, labels.reshape(-1, 1)], axis=1)
        sorted_point_label = raw_point_label[
            np.lexsort((grid_index[:, 0], grid_index[:, 1], grid_index[:, 2])), :
        ].astype(np.int64)
        voxel_label = label_voting(np.copy(voxel_label), sorted_point_label)

        # center points on voxel
        voxel_center = (grid_index.astype(float) + 0.5) * intervals + self.min_vol
        centered_xyz = xyz - voxel_center
        pt_features = np.concatenate((centered_xyz, xyz), axis=1)

        """
        *complete data_tuple*
        ---
        voxel_label: voxel-level label
        grid_index: individual point's grid index
        labels: individual point's label
        pt_features: [centered xyz, xyz, reflection]
        """
        # TODO: version data_tuple based on arguments

        if self.reflection:
            pt_features = np.concatenate((pt_features, reflection.reshape(-1, 1)), axis=1)

        return (voxel_label, grid_index, labels, pt_features)


@jit("u1[:,:,:](u1[:,:,:],i8[:,:])", nopython=True, cache=True, parallel=False)
def label_voting(voxel_label: np.array, sorted_list: list):

    # FIXME: way to similar to the original,
    # figure out how to do it differenlty (only do a lookup array for existing labels or sth)
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


# FIXME: this shit is needed to be able to have multiple instances with different sizes
def collate_fn_BEV(data):
    label2stack = np.stack([d[0] for d in data])
    grid_ind_stack = [d[1] for d in data]
    point_label = [d[2] for d in data]
    xyz = [d[3] for d in data]
    return torch.from_numpy(label2stack), grid_ind_stack, point_label, xyz


def main():

    # debugging polar_datamodule
    data_module = PolarNetDataModule(config_path="config/debug.yaml")
    data_module.setup()

    dataloader = data_module.val_dataloader()

    for data in dataloader:
        print(data)


if __name__ == "__main__":
    main()
