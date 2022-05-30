from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
from numba import jit
from torch.utils.data import DataLoader, Dataset

import src.misc.utils as utils


class PolarNetDataModule(pl.LightningDataModule):
    def __init__(self, config_name: str = "debug.yaml"):
        super().__init__()
        if Path("config/" + config_name).exists():
            self.config = utils.load_yaml("config/" + config_name)
        else:
            raise FileNotFoundError("Config file can not be found.")
        if Path(self.config["data_dir"]).exists():
            self.data_dir = self.config["data_dir"]
        else:
            raise FileNotFoundError("Data folder can not be found.")

        assert self.config["projection_type"] in ["polar", "cartesian", "spherical"], "incorrect projection type"

    def setup(self, stage: Optional[str] = None) -> None:

        # load the SemanticKITTI dataset
        if stage == "fit" or stage is None:
            self.semkitti_train = SemanticKITTI(self.data_dir, data_split="train")
            self.semkitti_valid = SemanticKITTI(self.data_dir, data_split="valid")
        if stage == "test" or stage is None:
            self.semkitti_test = SemanticKITTI(self.data_dir, data_split="test")

        # voxelize the datatset
        if stage == "fit" or stage is None:
            self.voxelised_train = voxelised_dataset(self.config, self.semkitti_train, data_split="train")
            self.voxelised_valid = voxelised_dataset(self.config, self.semkitti_valid, data_split="valid")
        if stage == "test" or stage is None:
            self.voxelised_test = voxelised_dataset(self.config, self.semkitti_test, data_split="test")

    def train_dataloader(self):
        return DataLoader(
            self.voxelised_train,
            collate_fn=collate_fn,
            shuffle=True,
            batch_size=self.config["train_batch"],
            num_workers=self.config["num_workers"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.voxelised_valid,
            collate_fn=collate_fn,
            shuffle=False,
            batch_size=self.config["valid_batch"],
            num_workers=self.config["num_workers"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.voxelised_test,
            collate_fn=collate_fn,
            shuffle=False,
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


class voxelised_dataset(Dataset):
    def __init__(
        self,
        config: dict,
        dataset,
        data_split,
    ):
        self.config = config
        self.dataset = dataset
        self.unlabeled_idx = utils.ignore_class(config["semkitti_config"])
        self.grid_size = np.asarray(config["grid_size"])
        self.max_vol = np.asarray(config["max_vol"], dtype=np.float32)
        self.min_vol = np.asarray(config["min_vol"], dtype=np.float32)
        self.data_split = data_split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        # extrat data
        data, labels = self.dataset[index]
        coordinate = data[:, :3]
        reflection = data[:, 3]

        if self.config["augmentations"]["flip"]:
            coordinate = utils.random_flip(coordinate)

        # random rotate
        if self.config["augmentations"]["rot"]:
            coordinate = utils.random_rot(coordinate)

        if self.config["projection_type"] == "polar":
            coordinate_xy = coordinate[:, :2].copy()  # copy 2 cartesian coordinates for the 7->9 features
            coordinate = utils.convert2polar(coordinate)

        # calculate the grid indices
        if self.config["augmentations"]["fixed_vol"]:
            coordinate = utils.clip(coordinate, self.min_vol, self.max_vol)
        else:
            self.max_vol = np.amax(coordinate, axis=0)
            self.min_vol = np.amin(coordinate, axis=0)

        step_size = (self.max_vol - self.min_vol) / (self.grid_size - 1)

        # calculate the grid index for each point
        grid_index = np.floor((coordinate - self.min_vol) / step_size).astype(int)

        # process the labels and vote for one per voxel #TODO: cite
        voxel_label = np.full(self.grid_size, self.unlabeled_idx, dtype=np.uint8)
        raw_point_label = np.concatenate([grid_index, labels.reshape(-1, 1)], axis=1)
        sorted_point_label = raw_point_label[
            np.lexsort((grid_index[:, 0], grid_index[:, 1], grid_index[:, 2])), :
        ].astype(np.int64)
        voxel_label = label_voting(np.copy(voxel_label), sorted_point_label)

        # center points on voxel # TODO: cite, not defined in the paper
        voxel_center = (grid_index.astype(float) + 0.5) * step_size + self.min_vol
        centered_coordinate = coordinate - voxel_center
        pt_features = np.concatenate((centered_coordinate, coordinate, reflection.reshape(-1, 1)), axis=1)
        if self.config["projection_type"] == "polar":
            pt_features = np.concatenate((pt_features, coordinate_xy), axis=1)

        if self.data_split == "test":
            voxelised_data = (voxel_label, grid_index, labels, pt_features, index)
        else:
            voxelised_data = (voxel_label, grid_index, labels, pt_features)

        """
        *complete data_tuple*
        ---
        voxel_label: voxel-level label
        grid_index: individual point's grid index
        labels: individual point's label
        pt_features: [centered coordinate, coordinate, reflection]
        """

        return voxelised_data


@jit("u1[:,:,:](u1[:,:,:],i8[:,:])", nopython=True, cache=True, parallel=False)
def label_voting(voxel_label: np.array, sorted_list: list):
    # TODO: cite
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


# to handle multi-size point clouds
def collate_fn(batch, test=False):
    label, grid_index, pt_label, pt_feature, index = [], [], [], [], []
    for i in batch:
        label.append(i[0])
        grid_index.append(i[1])
        pt_label.append(i[2])
        pt_feature.append(i[3])
        if test:
            index.append(i[4])

    collated = (np.stack(label), grid_index, pt_label, pt_feature)

    if test:
        collated += index

    return collated


def main():

    # debugging polar_datamodule
    data_module = PolarNetDataModule(config_name="debug.yaml")
    data_module.setup()

    dataloader = data_module.val_dataloader()

    for data in dataloader:
        print(data)


if __name__ == "__main__":
    main()
