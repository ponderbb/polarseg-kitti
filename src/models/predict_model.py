import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytorch_lightning as pl

BASE_DIR = os.path.abspath(os.curdir)

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.data.dataloader import PolarNetDataModule
from src.models.lightning_frame import PolarNetModule


def main(args):

    assert args.test or args.validate, "Neither test nor validation has been chosen. What did you expect?"

    # initialize trainer class
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=1, logger=False)

    # initialize data module
    polar_datamodule = PolarNetDataModule(args.config)

    # run the trained model instance on the validation set
    if args.validate:
        polar_datamodule.setup(stage="validate")
        polar_model = PolarNetModule(args.config, out_sequence=polar_datamodule.semkitti_valid)
        print("---\n Running inference on validation set:\n ---\n")
        trainer.validate(model=polar_model, datamodule=polar_datamodule)

    # generate labels from the test splits
    if args.test:
        polar_datamodule.setup(stage="test")
        polar_model = PolarNetModule(args.config, out_sequence=polar_datamodule.semkitti_test)
        print("---\n Running inference on test set:\n ---\n")
        trainer.test(model=polar_model, datamodule=polar_datamodule)

        if args.prep:
            """
            Prepare the generated labels for submission, based on the config.yaml file of the experiment.
            - remap labels based on documentation and script from [https://github.com/PRBonn/semantic-kitti-api]
            - zip label sequences
            - validate the zipped folder for submission based on documentation
            and scripts from [https://github.com/PRBonn/semantic-kitti-api]
            """

            print("---\n Preparing labels for submission\n ---\n")

            model_name = Path(polar_datamodule.config["model_save_path"]).stem
            remap_labels = (
                "python references/remap_semantic_labels.py -p models/inference/{}/ -s test --inverse -dc {}".format(
                    model_name, polar_datamodule.config["semkitti_config"]
                )
            )
            zip_to_folder = "(cd models/inference/{} && zip -r {}.zip sequences/)".format(model_name, model_name)
            validate_submission = (
                "python references/validate_submission.py --task segmentation models/inference/{}/{}.zip {}".format(
                    model_name, model_name, polar_datamodule.config["data_dir"]
                )
            )

            print("---\n Remapping labels\n ---\n")
            subprocess.call(remap_labels, shell=True)
            print("---\n Zipping to folder\n ---\n")
            subprocess.call(zip_to_folder, shell=True)
            print("---\n Validating submission zip folder\n ---\n")
            subprocess.call(validate_submission, shell=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="debug.yaml")
    parser.add_argument("--no-validate", dest="validate", action="store_false")
    parser.set_defaults(validate=True)
    parser.add_argument("--no-test", dest="test", action="store_false")
    parser.set_defaults(test=True)
    parser.add_argument("--no-prep", dest="prep", action="store_false")
    parser.set_defaults(prep=True)

    args = parser.parse_args()
    main(args)
