import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchlars import LARS

import src.misc.utils as utils
import wandb
from src.features.lovasz_losses import lovasz_softmax
from src.features.my_ptBEV import ptBEVnet


class PolarNetModule(pl.LightningModule):
    def __init__(self, config_name: str, out_sequence: Optional[Any] = None) -> None:
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
        self.out_sequence = out_sequence

    def setup(self, stage: Optional[str] = None) -> None:
        # setup logging
        if stage == "test" or stage == "validate":
            self.config["logging"] = False

        if self.config["logging"]:
            wandb.config.update(self.config)

        # load model
        self.model = ptBEVnet(
            backbone_name=self.config["backbone"],
            grid_size=self.config["grid_size"],
            projection_type=self.config["projection_type"],
            n_class=self.unique_class_idx,
            circular_padding=self.config["augmentations"]["circular_padding"],
            sampling=self.config["sampling"],
        )

        if stage == "validate" or stage == "test":
            if Path(self.config["model_save_path"]).exists():
                self.model.load_state_dict(torch.load(self.config["model_save_path"]))
            else:
                raise FileExistsError("No trained model found.")

        self.best_val_miou = 0
        self.exceptions = 0
        self.epoch = 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr_rate"])
        if self.config["LARS"]:
            optimizer = LARS(optimizer=optimizer, eps=1e-8, trust_coef=0.001)
        return optimizer

    def validation_step(self, batch, batch_idx):

        vox_label, grid_index, pt_label, pt_features = batch

        # remap labels from 0->255
        vox_label = utils.move_labels(vox_label, -1)
        pt_label = utils.move_labels(pt_label, -1)

        # convert things to tensors
        grid_index_tensor = [torch.from_numpy(i[:, :2]).type(torch.IntTensor).to(self.device) for i in grid_index]
        pt_features_tensor = [torch.from_numpy(i).type(torch.FloatTensor).to(self.device) for i in pt_features]
        vox_label_tensor = torch.from_numpy(vox_label).type(torch.LongTensor).to(self.device)

        prediction = self.model(pt_features_tensor, grid_index_tensor, self.device)

        cross_entropy_loss = self.loss_function(prediction.detach(), vox_label_tensor)
        lovasz_loss = lovasz_softmax(F.softmax(prediction).detach(), vox_label_tensor, ignore=255)
        combined_loss = lovasz_loss + cross_entropy_loss
        prediction = torch.argmax(prediction, dim=1)
        prediction = prediction.detach().cpu().numpy()
        for i, __ in enumerate(grid_index):
            self.hist_list.append(
                utils.fast_hist_crop(
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
        # TODO: cite miou
        iou = utils.per_class_iu(sum(self.hist_list))
        for class_name, class_iou in zip(self.unique_class_name, iou):
            if self.config["logging"]:
                wandb.log({f"{class_name}": class_iou})
            print("%s : %.2f%%" % (class_name, class_iou * 100))
        val_miou = np.nanmean(iou) * 100

        # save model if performance is improved
        if self.best_val_miou < val_miou:
            self.best_val_miou = val_miou
            torch.save(self.model.state_dict(), self.config["model_save_path"])
        print("---\nCurrent val_miou: {:.4f}\nBest val_miou: {:.4f}".format(val_miou, self.best_val_miou))

        if self.config["logging"]:
            wandb.log({"val_miou": val_miou, "best_val_miou": self.best_val_miou})

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
        vox_label = utils.move_labels(vox_label, -1)
        pt_label = utils.move_labels(pt_label, -1)

        grid_index_tensor = [torch.from_numpy(i[:, :2]).type(torch.IntTensor).to(self.device) for i in grid_index]
        pt_features_tensor = [torch.from_numpy(i).type(torch.FloatTensor).to(self.device) for i in pt_features]
        vox_label_tensor = torch.from_numpy(vox_label).type(torch.LongTensor).to(self.device)

        prediction = self.model(pt_features_tensor, grid_index_tensor, self.device)

        # NOTE: cite losses
        cross_entropy_loss = self.loss_function(prediction, vox_label_tensor)
        lovasz_loss = lovasz_softmax(F.softmax(prediction), vox_label_tensor, ignore=255)
        combined_loss = lovasz_loss + cross_entropy_loss

        if self.config["logging"]:
            wandb.log({"train_loss": combined_loss})
        self.loss_list.append(combined_loss.item())
        return combined_loss

    def on_test_start(self) -> None:

        self.inference_path = "models/inference/{}/".format(Path(self.config["model_save_path"]).stem)

        utils.inference_dir(self.inference_path)

    def test_step(self, batch, batch_idx):

        __, grid_index, __, pt_features, index = batch

        grid_index_tensor = [torch.from_numpy(i[:, :2]).type(torch.IntTensor).to(self.device) for i in grid_index]
        pt_features_tensor = [torch.from_numpy(i).type(torch.FloatTensor).to(self.device) for i in pt_features]

        prediction = self.model(pt_features_tensor, grid_index_tensor, self.device)
        prediction = (torch.argmax(prediction, 1)).cpu().detach().numpy()

        for i, __ in enumerate(grid_index):
            pt_pred_label = prediction[i, grid_index[i][:, 0], grid_index[i][:, 1], grid_index[i][:, 2]]
            pt_pred_label = utils.move_labels(pt_pred_label, 1)
            pt_id = Path(self.out_sequence.scan_list[index[i]]).stem
            assert pt_id == str(index[i]).zfill(6), "mismatch between load id: {} and write id: {}".format(
                pt_id, str(index[i]).zfill(6)
            )
            sequence = Path(self.out_sequence.scan_list[index[i]]).parents[1].stem
            new_file_path = "{}sequences/{}/predictions/{}.label".format(
                self.inference_path, sequence, str(pt_id).zfill(6)
            )

            if not Path(new_file_path).parents[0].exists():
                os.makedirs(Path(new_file_path).parents[0])
            pt_pred_label.tofile(new_file_path)
