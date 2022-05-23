import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

import src.misc.utils as utils
from src.data.dataloader import PolarNetDataModule
from src.features.lovasz_losses import lovasz_softmax
from src.features.my_BEV_Unet import BEV_Unet
from src.features.my_ptBEV import ptBEVnet


class PolarNetModule(pl.LightningModule):
    def __init__(self, config_path: str = "config/debug.yaml") -> None:
        super().__init__()

        # check if config path exists
        if Path(config_path).exists():
            self.config = utils.load_yaml(config_path)
        else:
            raise FileNotFoundError("Config file can not be found.")

        # load label information from semantic-kitti.yaml (api specific config)
        self.unique_classes = utils.load_SemKITTI_yaml(self.config["semkitti_config"], label_name=True)

        # # load models
        # self.BEV_model = BEV_Unet(n_class=len(self.unique_classes[0]), n_height=self.config["grid_size"][2])
        # self.model = ptBEVnet(self.device, self.BEV_model, self.config["grid_size"])

        # define variables based on config file
        self.model_path = self.config["model_save_path"]
        self.train_batch = self.config["train_batch"]
        self.valid_batch = self.config["valid_batch"]
        self.lr_rate = self.config["lr_rate"]
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=255)

    def setup(self, stage: Optional[str] = None) -> None:
        # load models

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # FIXME: trainer automatically does it but only after training loop start
        self.BEV_model = BEV_Unet(n_class=len(self.unique_classes[0]), n_height=self.config["grid_size"][2])
        self.model = ptBEVnet(self.BEV_model, self.config["grid_size"])
        self.best_val_miou = 0  # FIXME: make sure this is the correct place of definition
        self.exceptions = 0

    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def validation_step(self, batch, batch_idx):

        vox_label, grid_index, pt_label, pt_features = batch

        # remap labels from 0->255
        vox_label = utils.move_labels_back(vox_label)
        pt_label = utils.move_labels_back(pt_label)
        grid_index_tensor = [torch.from_numpy(i[:, :2]).to(self.device) for i in grid_index]
        pt_features = [torch.from_numpy(i).type(torch.FloatTensor).to(self.device) for i in pt_features]
        vox_label = vox_label.type(torch.LongTensor).to(self.device)

        prediction = self.model(
            pt_features, grid_index_tensor, circular_padding=False, device=self.device
        )  # TODO: what to do with the circular padding
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
                    self.unique_classes[0],
                )
            )
        self.val_loss_list.append(combined_loss.detach().cpu().numpy())

    # executes at the beggining of every evaluation
    def on_validation_start(self):
        self.val_loss_list = []
        self.hist_list = []

    # executes the per class iou calculations at the end of each validation block
    def on_validation_end(self):
        iou = per_class_iu(sum(self.hist_list))
        for class_name, class_iou in zip(self.unique_classes[1], iou):
            print("%s : %.2f%%" % (class_name, class_iou * 100))
        val_miou = np.nanmean(iou) * 100

        # save model if performance is improved
        if self.best_val_miou < val_miou:
            best_val_miou = val_miou
            torch.save(self.model.state_dict(), self.model_path)

        print("Current val miou is %.3f while the best val miou is %.3f" % (val_miou, best_val_miou))
        print("Current val loss is %.3f" % (np.mean(self.val_loss_list)))

    # initializations before new training
    def on_train_start(self) -> None:
        self.loss_list = []

    def training_step(self, batch, batch_idx):
        # try:
        vox_label, grid_index, pt_label, pt_features = batch

        # remap labels from 0->255
        vox_label = utils.move_labels_back(vox_label)
        pt_label = utils.move_labels_back(pt_label)
        grid_index_tensor = [torch.from_numpy(i[:, :2]).to(self.device) for i in grid_index]
        pt_features = [torch.from_numpy(i).type(torch.FloatTensor).to(self.device) for i in pt_features]
        vox_label = vox_label.type(torch.LongTensor).to(self.device)

        prediction = self.model(
            pt_features, grid_index_tensor, circular_padding=False, device=self.device
        )  # TODO: what to do with the circular padding
        cross_entropy_loss = self.loss_function(prediction, vox_label)
        lovasz_loss = lovasz_softmax(F.softmax(prediction), vox_label, ignore=255)
        combined_loss = lovasz_loss + cross_entropy_loss
        self.log("train_loss", combined_loss)
        self.loss_list.append(combined_loss.item())
        return combined_loss
        # except Exception as error:
        #     if self.exceptions == 0:
        #         print(error)
        #     self.exceptions += 1


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(n * label[k].astype(int) + pred[k], minlength=n**2)
    return bin_count[: n**2].reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 1)
    hist = hist[unique_label, :]
    hist = hist[:, unique_label]
    return hist


def main(args):

    polar_datamodule = PolarNetDataModule(args.config)
    polar_model = PolarNetModule(args.config)
    trainer = pl.Trainer(val_check_interval=100, accelerator="gpu", devices=1)

    trainer.fit(model=polar_model, datamodule=polar_datamodule)

    # Trainer -> val_check_interval = 0.25


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config/debug.yaml")

    args = parser.parse_args()
    print(args)
    main(args)
