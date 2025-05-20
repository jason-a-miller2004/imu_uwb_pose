# test.py

import os
import argparse
import pytorch_lightning as pl
import torch
import smplx
import pickle

from imu_uwb_pose import config as c, eval_metrics as e
from imu_uwb_pose.training.imu_uwb_model import imu_uwb_pose_model
from imu_uwb_pose.training.utils import imu_uwb_data_module as imu_uwb_data_module
import numpy as np


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
        type=str,
        help="If set, the script will run in fine-tuning mode (optional)."
    )

    parser.add_argument(
        "--lr",
        type=str,
        required=True,
        help="Current learning rate being tested"
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 2) Initialize config with the provided experiment name
    # -------------------------------------------------------------------------

    # Optional: if --finetune is set
    if args.finetune:
        config = c.config(
            experiment=args.experiment,
            dataset="footposer_dataset",
            name=args.finetune
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
        devices = [0]
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

    body_model = smplx.create(config.body_model, model_type='smplx',
                         gender='neutral', use_face_contour=False,
                         batch_size=1,
                         ext='npz',
                         age='adult').to(config.device)

    err_dict = e.get_metrics(outputs, body_model, config)

    angle_err = np.mean(err_dict['angle_error'])
    joint_err = np.mean(err_dict['joint_error'])
    vertex_err = np.mean(err_dict['vertex_error'])
    jitter_err = np.mean(err_dict['jitter'])

    print(f'{args.finetune} results')
    print(f'angle error {angle_err}')
    print(f'joint error {joint_err}')
    print(F'vertex error {vertex_err}')
    print(f'jitter_err {jitter_err}')
    


    save_dir = os.path.relpath(f"./data/results/LOO_{args.lr}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"{args.finetune}_error_metrics.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(err_dict, f)

    print("Saved errors to " + save_path)


