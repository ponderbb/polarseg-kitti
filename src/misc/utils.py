import os
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


def load_SemKITTI_yaml(file, label_name=False):
    # FIXME: rewrite or reference

    with open(file, "r") as stream:
        semkitti_dict = yaml.safe_load(stream)

    # data tuple for unique label values and keys
    if label_name:
        unique_dict = dict()
        for i in sorted(list(semkitti_dict["learning_map"].keys()))[::-1]:
            unique_dict[semkitti_dict["learning_map"][i]] = semkitti_dict["labels"][i]

        unique_keys = np.asarray(sorted(list(unique_dict.keys())))[1:] - 1
        unique_labels = [unique_dict[x] for x in unique_keys + 1]

        return (unique_keys, unique_labels)

    return semkitti_dict


def load_yaml(file):
    with open(file, "r") as stream:
        dict = yaml.safe_load(stream)
    return dict


def move_labels_back(label):
    if isinstance(label, list):
        return [i - 1 for i in label]
    else:
        return label - 1
