import os
from pathlib import Path

import yaml
from torch.utils.data import Dataset


class SemanticKITTI(Dataset):
    def __init__(self, path, data_split="train") -> None:
        self.data_split = data_split
        self.image_list = []

        with open("semantic-kitti.yaml", "r") as stream:
            dataset_config = yaml.safe_load(stream)

        try:
            split = dataset_config["split"][self.data_split]
            print(split)
        except ValueError:
            print("Incorrect set type")

        for sequence_folder in split:
            self.image_list += getPath(
                "/".join([path, "sequences", str(sequence_folder).zfill(2), "velodyne"])
            )
        self.image_list.sort()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        return None


def getPath(dir):
    for root, dirs, files in os.walk(dir):
        for f in files:
            yield str(Path(os.path.join(root, f)))


def main():

    train_dataset = SemanticKITTI(
        path="/root/repos/polarseg-kitti/data/debug", data_split="train"
    )
    print(train_dataset)


if __name__ == "__main__":
    main()
