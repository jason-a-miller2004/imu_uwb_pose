# %%
# %load_ext autoreload
# %autoreload 2

# %%
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import seed_everything

from imu_uwb_pose import config as c
from imu_uwb_pose.training import imu_uwb_model as model
from imu_uwb_pose.training.utils import imu_uwb_data_module as imu_uwb_data_module


config = c.config(experiment='imu_uwb_pose_amass', dataset='amass_dataset')

if __name__ == "__main__":
    # set the random seed
    seed_everything(config.torch_seed, workers=True)

    # instantiate model and data
    model = model.imu_uwb_pose_model(config)
    datamodule = imu_uwb_data_module(config)
    checkpoint_path = config.checkpoint_path 

    # set up WandB logger
    wandb_logger = WandbLogger(project=config.experiment, save_dir=checkpoint_path)

    early_stopping_callback = EarlyStopping(monitor="validation_step_loss", mode="min", verbose=False,
                                            min_delta=0.00001, patience=5)
    checkpoint_callback = ModelCheckpoint(monitor="validation_step_loss", mode="min", verbose=False, 
                                          save_top_k=5, dirpath=checkpoint_path, save_weights_only=True, 
                                          filename='epoch={epoch}-val_loss={validation_step_loss:.5f}')

    if config.device.type == 'cuda':
        accelerator = "gpu"
        devices = [0]
    else:
        accelerator = "cpu"
        devices = 1

    trainer = pl.Trainer(fast_dev_run=False, logger=wandb_logger, max_epochs=1000, accelerator=accelerator, devices=devices,
                         callbacks=[early_stopping_callback, checkpoint_callback], deterministic=True)

    trainer.fit(model, datamodule=datamodule)

    with open(checkpoint_path / "best_model.txt", "w") as f:
        f.write(f"{checkpoint_callback.best_model_path}\n\n{checkpoint_callback.best_k_models}")