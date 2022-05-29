import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger

import wandb

BASE_DIR = os.path.abspath(os.curdir)
print(BASE_DIR)

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import src.misc.utils as utils
from src.data.dataloader import PolarNetDataModule
from src.features.lovasz_losses import lovasz_softmax
from src.features.my_ptBEV import ptBEVnet


class PolarNetModule(pl.LightningModule):
    def __init__(self, config_name: str = "debug.yaml") -> None:
        super().__init__()

        # check if config path exists
        if Path("config/" + config_name).exists():
            self.config = utils.load_yaml("config/" + config_name)
        else:
            raise FileNotFoundError("Config file can not be found.")

        # load label information from semantic-kitti.yaml (api specific config)
        self.unique_class_idx, self.unique_class_name = utils.load_unique_classes(self.config["semkitti_config"])

        # define variables based on config file
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=255)

    def setup(self, stage: Optional[str] = None) -> None:
        # setup logging

        if self.config["logging"]:
            wandb.config.update(self.config)

        # load models
        self.model = ptBEVnet(
            backbone=self.config["backbone"],
            grid_size=self.config["grid_size"],
            projection_type=self.config["projection_type"],
            n_class=self.unique_class_idx,
            circular_padding=self.config["augmentations"]["circular_padding"],
            device = self.device,
        )
        self.best_val_miou = 0
        self.exceptions = 0
        self.epoch = 0

    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters(), lr=self.config["lr_rate"])

    def validation_step(self, batch, batch_idx):

        vox_label, grid_index, pt_label, pt_features = batch

        # remap labels from 0->255
        vox_label = utils.move_labels_back(vox_label)
        pt_label = utils.move_labels_back(pt_label)
        grid_index_tensor = [torch.from_numpy(i[:, :2]).to(self.device) for i in grid_index]
        pt_features = [torch.from_numpy(i).type(torch.FloatTensor).to(self.device) for i in pt_features]
        vox_label = torch.from_numpy(vox_label).type(torch.LongTensor).to(self.device)

        prediction = self.model(
            pt_features,
            grid_index_tensor,
        )

        cross_entropy_loss = self.loss_function(prediction.detach(), vox_label)
        lovasz_loss = lovasz_softmax(F.softmax(prediction).detach(), vox_label, ignore=255)
        combined_loss = lovasz_loss + cross_entropy_loss
        prediction = torch.argmax(prediction, dim=1)
        prediction = prediction.detach().cpu().numpy()
        for i, __ in enumerate(grid_index):
            self.hist_list.append(
                fast_hist_crop(
                    prediction[i, grid_index[i][:, 0], grid_index[i][:, 1], grid_index[i][:, 2]],
                    pt_label[i],
                    self.unique_class_idx,
                )
            )
        if self.config["logging"]:
            wandb.log({"val_loss": combined_loss})
        self.val_loss_list.append(combined_loss.detach().cpu().numpy())

    # executes at the beggining of every evaluation
    def on_validation_start(self):
        self.val_loss_list = []
        self.hist_list = []

    # executes the per class iou calculations at the end of each validation block
    def on_validation_end(self):
        iou = per_class_iu(sum(self.hist_list))
        for class_name, class_iou in zip(self.unique_class_name, iou):
            if self.config["logging"]:
                wandb.log({f"{class_name}": class_iou})
            print("%s : %.2f%%" % (class_name, class_iou * 100))
        val_miou = np.nanmean(iou) * 100

        # save model if performance is improved
        if self.best_val_miou < val_miou:
            self.best_val_miou = val_miou
            torch.save(self.model.state_dict(), self.config["model_save_path"])

        if self.config["logging"]:
            wandb.log({"val_miou": val_miou, "best_val_miou": self.best_val_miou})

        print("Current val miou is %.3f while the best val miou is %.3f" % (val_miou, self.best_val_miou))

    # initializations before new training
    def on_train_start(self) -> None:
        self.loss_list = []

    def on_train_epoch_start(self) -> None:
        self.epoch += 1
        if self.config["logging"]:
            wandb.log({"epoch": self.epoch})

    def training_step(self, batch, batch_idx):

        vox_label, grid_index, pt_label, pt_features = batch

        # remap labels from 0->255
        vox_label = utils.move_labels_back(vox_label)
        pt_label = utils.move_labels_back(pt_label)
        grid_index_tensor = [torch.from_numpy(i[:, :2]).to(self.device) for i in grid_index]
        pt_features = [torch.from_numpy(i).type(torch.FloatTensor).to(self.device) for i in pt_features]
        vox_label = torch.from_numpy(vox_label).type(torch.LongTensor).to(self.device)

        prediction = self.model(
            pt_features,
            grid_index_tensor,
        )

        cross_entropy_loss = self.loss_function(prediction, vox_label)
        lovasz_loss = lovasz_softmax(F.softmax(prediction), vox_label, ignore=255)
        combined_loss = lovasz_loss + cross_entropy_loss

        if self.config["logging"]:
            wandb.log({"train_loss": combined_loss})
        self.loss_list.append(combined_loss.item())
        return combined_loss


# TODO: cite or replace
def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(n * label[k].astype(int) + pred[k], minlength=n**2)
    return bin_count[: n**2].reshape(n, n)


# TODO: cite or replace
def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


# TODO: cite or replace
def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 1)
    hist = hist[unique_label, :]
    hist = hist[:, unique_label]
    return hist


def main(args):

    polar_datamodule = PolarNetDataModule(args.config)
    polar_model = PolarNetModule(args.config)
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
    )

    trainer.fit(model=polar_model, datamodule=polar_datamodule)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="debug.yaml")

    args = parser.parse_args()
    main(args)
