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

def load_SemKITTI_yaml(file, label_name=True):

    with open(file, "r") as stream:
        semkitti_dict = yaml.safe_load(stream)

    # replaces learning_map values with name strings instead of numbers
    if label_name:
        for i in list(semkitti_dict['learning_map'].keys()):
            semkitti_dict['learning_map'][i] = semkitti_dict['labels'][i]

    return semkitti_dict
