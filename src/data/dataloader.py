import os
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


class SemanticKITTI(Dataset):
    def __init__(self, path, data_split="train", reflection=True) -> None:

        with open("semantic-kitti.yaml", "r") as stream:
            dataset_yaml = yaml.safe_load(stream)

        self.data_split = data_split
        self.reflection = reflection
        self.learning_map = dataset_yaml["learning_map"]
        self.scan_list = []
        self.label_list = []
        try:
            split = dataset_yaml["split"][self.data_split]
            print(split)
        except ValueError:
            print("Incorrect set type")

        for sequence_folder in split:
            self.scan_list += getPath("/".join([path, "sequences", str(sequence_folder).zfill(2), "velodyne"]))
            self.label_list += getPath("/".join([path, "sequences", str(sequence_folder).zfill(2), "labels"]))
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


def getPath(dir):
    for root, _, files in os.walk(dir):
        for f in files:
            yield str(Path(os.path.join(root, f)))


def main():

    train_dataset = SemanticKITTI(path="/root/repos/polarseg-kitti/data/debug", data_split="test")
    dummy_dataloader = torch.utils.data.DataLoader(train_dataset)

    for data_tuple in dummy_dataloader:
        print(data_tuple[0])
        print(data_tuple[1])


if __name__ == "__main__":
    main()
