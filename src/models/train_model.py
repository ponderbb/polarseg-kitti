import argparse
import os
import string
import typing
from ast import Str
from cProfile import label
from pathlib import Path
from syslog import LOG_SYSLOG

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.features.my_BEV_Unet import BEV_Unet
from src.features.my_ptBEV import ptBEVnet
from src.misc import utils


class PolarNetTrainer(pl.LightningModule):
    def __init__(self, config_path: str = "config/debug.yaml") -> None:
        super().__init__()
        self.BEV_model = BEV_Unet()
        self.ptBEV_model = ptBEVnet()
        if Path(config_path).exists():
            self.config = utils.load_yaml(config_path)
        else:
            raise FileNotFoundError("Config file can not be found.")
        self.model_path = self.config["model_save_path"]
        self.train_batch = self.config["train_batch"]
        self.valid_batch = self.config["valid_batch"]
        self.lr_rate = self.config["lr_rate"]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def training_step(self, batch, batch_idx):
        vox_label, grid_index, pt_label, pt_features = batch
        # TODO: transform the labels 0->255

        return loss


def main(args):

    semkitti_dict = utils.load_SemKITTI_yaml("semantic-kitti.yaml", label_name=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", default="data")
    parser.add_argument("-p", "--model_out_path", default="./my_SemKITTI_traditionalSeg.pt")
    parser.add_argument(
        "-v",
        "--BEV",
        choices=["polar", "traditional"],
        default="traditional",
        help="model version: polar or traditional",
    )
    parser.add_argument("-s", "--grid_size", nargs="+", type=int, default=[480, 360, 32])
    parser.add_argument("--debug", type=bool, default=True)

    args = parser.parse_args()
    print(args)
    main(args)
