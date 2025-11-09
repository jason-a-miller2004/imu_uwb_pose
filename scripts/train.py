# %%
# %load_ext autoreload
# %autoreload 2

# %%
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import seed_everything

import argparse
import importlib

from imu_uwb_pose import config as c
from imu_uwb_pose.training.utils import imu_uwb_data_module as imu_uwb_data_module
from imu_uwb_pose.training.utils import get_model
from imu_uwb_pose.training.imu_uwb_pose_model import imu_uwb_pose_model
from pathlib import Path

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 1) Parse command-line arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Train an IMU UWB Pose model.")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Name of the experiment to run. Corresponds to config.experiment."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model to use"
    )
    parser.add_argument(
        "--loo",
        type=str,
        required=False,
        help="test subject to leave out if doing loo cross validation"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of the dataset to use"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-3,
        help="Learning rate for the optimizer."
    )

    parser.add_argument(
        "--finetune",
        type=str,
        required=False
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 2) Use experiment argument in config
    # -------------------------------------------------------------------------
    config = c.config(
            experiment=args.experiment,
            dataset=args.dataset,
            lr=args.lr,
            name=args.loo
    )
    experiment = config.experiment
    checkpoint_path = config.checkpoint_path

    # Optionally, do something special if --finetune was set:
    if args.finetune:
        # load the checkpoint path
        pretrain_model_path = config.imu_uwb_pose_model_path / f"pose_models/checkpoints/{args.finetune}"
        # read first line of best model.txt
        with open(pretrain_model_path / "best_model.txt", "r") as f:
            lines = f.readlines()
            pretrain_model_path = lines[0].strip()

        # load the model
        model = imu_uwb_pose_model.load_from_checkpoint(
            pretrain_model_path,
            config=config,
            map_location=config.device
        )
    else:
        model = imu_uwb_pose_model(config, get_model(config,args.model))



    # set the random seed
    seed_everything(config.torch_seed, workers=True)

    # instantiate model and data
    datamodule = imu_uwb_data_module(config)

    # set up WandB logger
    wandb_logger = WandbLogger(project=experiment, save_dir=checkpoint_path)

    # early_stopping_callback = EarlyStopping(
    #     monitor="validation_step_loss",
    #     mode="min",
    #     verbose=False,
    #     min_delta=0.00001,
    #     patience=5
    # )
    checkpoint_callback = ModelCheckpoint(
        monitor="validation_step_loss",
        mode="min",
        verbose=False,
        save_top_k=2,
        dirpath=checkpoint_path,
        save_weights_only=True,
        filename='epoch={epoch}-val_loss={validation_step_loss:.5f}'
    )

    print('config device type ', config.device.type)
    if config.device.type == 'cuda':
        accelerator = "gpu"
        print('using gpu')
        devices = [0]
    else:
        accelerator = "cpu"
        print('using cpu')
        devices = 1

    trainer = pl.Trainer(
        fast_dev_run=False,
        logger=wandb_logger,
        max_epochs=10,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback],
        deterministic=True
    )

    # log initial validation loss
    trainer.validate(model, datamodule=datamodule)

    trainer.fit(model, datamodule=datamodule)

    with open(checkpoint_path / "best_model.txt", "w") as f:
        f.write(f"{checkpoint_callback.best_model_path}\n\n{checkpoint_callback.best_k_models}")
