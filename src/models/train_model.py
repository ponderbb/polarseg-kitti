from ast import Str
from cProfile import label
import string
from syslog import LOG_SYSLOG
import torch 
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse
from src.features.my_BEV_Unet import BEV_Unet
from src.features.my_ptBEV import ptBEVnet
from src.misc import utils
from pathlib import Path
import typing


class PolarNetTrainer(pl.LightningModule):
    def __init__(self,
        data_path: typing.Union[str,Path] = '/data/',
        model_path: typing.Union[str,Path] = '/data',
        model_type: str = 'traditional',
        grid_size: list[int] = [480,360,32],
        train_batch: int = 2,
        valid_batch: int = 2,
        lr_rate: float = 2e-2) -> None:
        super().__init__()
        self.BEV_model = BEV_Unet()
        self.ptBEV_model = ptBEVnet()
        self.data_path = data_path
        self.model_path = model_path
        self.model_type = model_type
        self.grid_size = grid_size
        self.train_batch = train_batch
        self.valid_batch = valid_batch
        self.lr_rate = lr_rate


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def train_dataloader(self) -> TRAIN_DATALOADERS:

        

        return super().train_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return super().val_dataloader()

    def training_step(self, batch, batch_idx):
        vox_label, grid_index, pt_label, pt_features = batch
        # TODO: transform the labels 0->255
        
        return loss


def main(args):

    semkitti_dict = utils.load_SemKITTI_yaml("semantic-kitti.yaml", label_name=True)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='data')
    parser.add_argument('-p', '--model_out_path', default='./my_SemKITTI_traditionalSeg.pt')
    parser.add_argument('-v', '--BEV', choices=['polar','traditional'], default='traditional', help='model version: polar or traditional')
    parser.add_argument('-s', '--grid_size', nargs='+', type=int, default = [480,360,32])
    parser.add_argument('--debug', type=bool, default=True)

    args = parser.parse_args()
    print(args)
    main(args)
