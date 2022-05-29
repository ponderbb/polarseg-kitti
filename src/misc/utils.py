import os
import random
from pathlib import Path

import numpy as np
import yaml


def clip(input, min_bound, max_bound, out_type=int):
    assert (min_bound < max_bound).all(), "lower and upper volume boundary mismatching"
    return np.minimum(max_bound, np.maximum(input, min_bound))


def getPath(dir):
    for root, _, files in os.walk(dir):
        for f in files:
            yield str(Path(os.path.join(root, f)))


def load_unique_classes(semkitti_yaml):

    semkitti_dict = load_yaml(semkitti_yaml)

    labels = np.fromiter(semkitti_dict["learning_map"].values(), dtype=np.int8)
    class_idx = np.unique(labels)[:-1]  # shift by, one extra due to [unlabelled]
    class_name = []

    for i in class_idx:
        class_name.append(semkitti_dict["labels"][semkitti_dict["learning_map_inv"][i + 1]])

    return class_idx, class_name


def ignore_class(semkitti_yaml):
    """
    based on semantickitti api, defines the "unlabelled instance (0)
    """
    semkitti_dict = load_yaml(semkitti_yaml)
    for key, value in semkitti_dict["learning_ignore"].items():
        if value:
            return key


def remap_labels(labels, semkitti_dict):
    """
    based on semantickitti api, remap labels to cross-entropy form
    """
    new_labels = np.zeros(len(labels), dtype=np.int8)
    for idx, lab in enumerate(labels.squeeze()):
        new_labels[idx] = semkitti_dict["learning_map"][lab]
    return new_labels


def load_yaml(file):
    with open(file, "r") as stream:
        dict = yaml.safe_load(stream)
    return dict


def move_labels_back(label):
    if isinstance(label, list):
        return [i - 1 for i in label]
    else:
        return label - 1


def random_flip(xyz):
    choice = random.randint(0, 3)
    if choice == 1:
        # print("Flip along x-axis")
        xyz[:, 0] = np.negative(xyz[:, 0])
    elif choice == 2:
        # print("Flip along y-axis")
        xyz[:, 1] = np.negative(xyz[:, 1])
    elif choice == 3:
        # print("Flip along both axis")
        xyz[:, 0] = np.negative(xyz[:, 0])
        xyz[:, 1] = np.negative(xyz[:, 1])
    return xyz


def random_rot(xyz):
    angle = np.random.randint(0, 360)
    x = (xyz[:, 0] * np.cos(np.deg2rad(angle)) - xyz[:, 1] * np.sin(np.deg2rad(angle))).reshape(-1, 1)
    y = (xyz[:, 1] * np.cos(np.deg2rad(angle)) + xyz[:, 0] * np.sin(np.deg2rad(angle))).reshape(-1, 1)
    z = xyz[:, 2].reshape(-1, 1)
    return np.concatenate((x, y, z), axis=1)
