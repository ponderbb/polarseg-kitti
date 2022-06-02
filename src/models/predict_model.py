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

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=1,
    )

    if args.validate:

        polar_datamodule = PolarNetDataModule(args.config)
        polar_model = PolarNetModule(args.config, out_sequence=None)

        trainer.validate(model=polar_model, datamodule=polar_datamodule)
    else:

        polar_datamodule.setup(stage="test")
        polar_model = PolarNetModule(args.config, out_sequence=polar_datamodule.semkitti_test)
        trainer.test(model=polar_model, datamodule=polar_datamodule)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="debug.yaml")
    parser.add_argument(
        "-v",
        "--validate",
        type=bool,
        default=True,
        help="Running inference on validation set, before saving the test labels.",
    )

    args = parser.parse_args()
    main(args)
