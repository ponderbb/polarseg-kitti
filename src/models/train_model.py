import argparse
import os
import sys

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

BASE_DIR = os.path.abspath(os.curdir)

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.data.dataloader import PolarNetDataModule
from src.models.lightning_frame import PolarNetModule


def main(args):

    polar_datamodule = PolarNetDataModule(args.config)
    polar_model = PolarNetModule(args.config, out_sequence=None)
    if polar_model.config["logging"]:
        logger = WandbLogger(project=polar_model.config["wandb_project"], log_model="True", entity="cs492_t13")
    else:
        logger = None

    trainer = pl.Trainer(
        val_check_interval=polar_model.config["val_check_interval"],
        accelerator="gpu",
        devices=1,
        logger=logger,
        default_root_dir="models/",
        max_epochs=polar_model.config["max_epochs"],
    )

    trainer.fit(model=polar_model, datamodule=polar_datamodule)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="debug.yaml")

    args = parser.parse_args()
    main(args)
