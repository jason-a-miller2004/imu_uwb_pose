# test.py

import os
import argparse
import pytorch_lightning as pl
import torch

from imu_uwb_pose import config as c
from imu_uwb_pose.training.imu_uwb_model import imu_uwb_pose_model
from imu_uwb_pose.training.utils import imu_uwb_data_module as imu_uwb_data_module


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 1) Parse command-line arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Test an IMU UWB Pose model.")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Name of the experiment to run. Corresponds to config.experiment."
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="If set, the script will run in fine-tuning mode (optional)."
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 2) Initialize config with the provided experiment name
    # -------------------------------------------------------------------------

    # Optional: if --finetune is set, do something special here
    if args.finetune:
        config = c.config(
            experiment=args.experiment + "-finetune",
            dataset="footposer_dataset"
        )
    else:
        config = c.config(
            experiment=args.experiment,
            dataset="amass_dataset"
        )

    # -------------------------------------------------------------------------
    # 3) Read the best model path from best_model.txt
    # -------------------------------------------------------------------------
    best_model_txt_path = os.path.join(config.checkpoint_path, "best_model.txt")
    with open(best_model_txt_path, "r") as f:
        lines = f.readlines()
    best_model_path = lines[0].strip()

    print(f"Loading the best model from: {best_model_path}")

    # -------------------------------------------------------------------------
    # 4) Load the best model
    #    Note: map_location is set to config.device (e.g., CPU or GPU).
    # -------------------------------------------------------------------------
    model = imu_uwb_pose_model.load_from_checkpoint(
        best_model_path,
        map_location=config.device,
        config=config
    )

    # -------------------------------------------------------------------------
    # 5) Instantiate the data module
    # -------------------------------------------------------------------------
    datamodule = imu_uwb_data_module(config)

    # -------------------------------------------------------------------------
    # 6) Create a trainer (configure GPU or CPU as needed)
    # -------------------------------------------------------------------------
    if config.device.type == 'cuda':
        accelerator = "gpu"
        devices = [6]
        print("Testing on GPU...")
    else:
        accelerator = "cpu"
        devices = 1
        print("Testing on CPU...")

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        fast_dev_run=False,
    )

    print("Running model predictions on the test set...")
    outputs = trainer.predict(model, datamodule=datamodule)

    torch.save(outputs, "outputs.pt")

    print("Predictions saved to 'outputs.pt'.")
    print("Test run complete.")


