import json
import os
import random
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import confusion_matrix

np.seterr(invalid="ignore")  # resolve true divide warning


def rebase(input, min_bound, max_bound, out_type=int):
    """
    limit volume space of point cloud to ROI
    """
    assert (min_bound < max_bound).all(), "lower and upper volume boundary mismatching"
    limited_coordinate = np.minimum(max_bound, np.maximum(input, min_bound))
    rebase_to_zero = limited_coordinate - min_bound
    return rebase_to_zero


def write_dict(file: dict, out_path: str):
    """
    dump dictionary into .txt file
    """
    with open(out_path, "w") as out_file:
        out_file.write(json.dumps(file))
        out_file.close()


def getPath(dir):
    """
    get full path for files in passed folder
    """
    for root, _, files in os.walk(dir):
        for f in files:
            yield str(Path(os.path.join(root, f)))


def load_unique_classes(semkitti_yaml):
    """
    load the unique classes, based on the remapped unique labels
    """
    semkitti_dict = load_yaml(semkitti_yaml)

    labels = np.fromiter(semkitti_dict["learning_map"].values(), dtype=np.int8)
    class_idx = np.unique(labels)[:-1]  # shift by, one extra due to [unlabelled]
    class_name = []

    for i in class_idx:
        class_name.append(semkitti_dict["labels"][semkitti_dict["learning_map_inv"][i + 1]])

    return class_idx, class_name


def ignore_class(semkitti_yaml):
    """
    based on [https://github.com/PRBonn/semantic-kitti-api]
    defines the "unlabelled instance (0)
    """
    semkitti_dict = load_yaml(semkitti_yaml)
    for key, value in semkitti_dict["learning_ignore"].items():
        if value:
            return key


def remap_labels(labels, semkitti_dict):
    """
    based on [https://github.com/PRBonn/semantic-kitti-api]
    remap labels to cross-entropy form
    """
    new_labels = np.zeros(len(labels), dtype=np.int8)
    for idx, lab in enumerate(labels.squeeze()):
        new_labels[idx] = semkitti_dict["learning_map"][lab]
    return new_labels


def load_yaml(file):
    with open(file, "r") as stream:
        dict = yaml.safe_load(stream)
    return dict


def move_labels(label, n):
    """
    moving unassigned labels from 0 -> 255 for unknown reason
    according to [https://github.com/edwardzhou130/PolarSeg]
    """
    if isinstance(label, list):
        return [i + np.uint8(n) for i in label]
    else:
        return label + np.uint8(n)


def random_flip(xyz):
    """
    flip input point cloud randomly along x-, y- or both axes
    """
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
    """
    Source equation for the coordinate roation around the origo (z-axis):
    https://doubleroot.in/lessons/coordinate-geometry-basics/rotation-of-axes/
    """
    angle = np.random.randint(0, 360)
    x = (xyz[:, 0] * np.cos(np.deg2rad(angle)) - xyz[:, 1] * np.sin(np.deg2rad(angle))).reshape(-1, 1)
    y = (xyz[:, 1] * np.cos(np.deg2rad(angle)) + xyz[:, 0] * np.sin(np.deg2rad(angle))).reshape(-1, 1)
    z = xyz[:, 2].reshape(-1, 1)
    return np.concatenate((x, y, z), axis=1)


def convert2polar(xyz):
    """
    Source equation for the cartesian to polar conversion:
    https://brilliant.org/wiki/convert-cartesian-coordinates-to-polar/
    """
    r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    theta = np.arctan2(xyz[:, 1], xyz[:, 0])
    z = xyz[:, 2]
    return np.stack((r, theta, z), axis=1)


def conf_mat_generator(prediction, label, classes, ignore_class):
    """
    generate keras [classes, classes] dimensional confusion matrix,
    based on the results with valid labels (ignore = 255)
    """
    ignore_idx = label != ignore_class
    return confusion_matrix(label[ignore_idx], prediction[ignore_idx], labels=classes)


def class_iou(cm):
    """
    Calculate intersection over union from confusion matrix, where diagonal holds the true positives (TP).
    IoU = TP / (TP+FP+FN) -> summing along the axis, we get the TP twice in the denominator!
    """
    intersection = cm.diagonal()
    union = cm.sum(0) - cm.diagonal() + cm.sum(1)
    return (intersection / union) * 100


def inference_dir(inference_path=str):
    """
    initialize empty directory for output labels, with the name of the model
    """
    os.makedirs(inference_path, exist_ok=True)


def collate_fn(batch):
    label, grid_index, pt_label, pt_feature, index = [], [], [], [], []
    for i in batch:
        label.append(i[0].astype(np.uint8))
        grid_index.append(i[1])
        pt_label.append(i[2].astype(np.uint8))
        pt_feature.append(i[3])
        index.append(i[4])

    collated = (np.stack(label), grid_index, pt_label, pt_feature, index)

    return collated
