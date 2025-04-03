# %%
# %load_ext autoreload
# %autoreload 2

# %%
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import seed_everything

import argparse

from imu_uwb_pose import config as c
from imu_uwb_pose.training import imu_uwb_model as model
from imu_uwb_pose.training.utils import imu_uwb_data_module as imu_uwb_data_module

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
        "--finetune",
        action="store_true",
        help="If set, the script will run in fine-tuning mode (optional)."
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 2) Use experiment argument in config
    # -------------------------------------------------------------------------
    config = c.config(
        experiment=args.experiment,
        dataset="amass_dataset"  # or read from another cmd arg if you prefer
    )
    
    # Optionally, do something special if --finetune was set:
    if args.finetune:
        print("Fine-tuning mode is ON.")

    # set the random seed
    seed_everything(config.torch_seed, workers=True)

    # instantiate model and data
    model = model.imu_uwb_pose_model(config)
    datamodule = imu_uwb_data_module(config)
    checkpoint_path = config.checkpoint_path 

    # set up WandB logger
    wandb_logger = WandbLogger(project=config.experiment, save_dir=checkpoint_path)

    early_stopping_callback = EarlyStopping(
        monitor="validation_step_loss",
        mode="min",
        verbose=False,
        min_delta=0.00001,
        patience=5
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="validation_step_loss",
        mode="min",
        verbose=False,
        save_top_k=5,
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
        max_epochs=1000,
        accelerator=accelerator,
        devices=devices,
        callbacks=[early_stopping_callback, checkpoint_callback],
        deterministic=True
    )

    # If --finetune is set, you might, for example, load a checkpoint:
    # if args.finetune:
    #     model.load_state_dict(torch.load("some_checkpoint.ckpt"))

    trainer.fit(model, datamodule=datamodule)

    with open(checkpoint_path / "best_model.txt", "w") as f:
        f.write(f"{checkpoint_callback.best_model_path}\n\n{checkpoint_callback.best_k_models}")
