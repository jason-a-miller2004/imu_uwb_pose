import os
import pytorch_lightning as pl
import torch

from imu_uwb_pose import config as c
from imu_uwb_pose.training.imu_uwb_model import imu_uwb_pose_model
from imu_uwb_pose.training.utils import imu_uwb_data_module as imu_uwb_data_module


if __name__ == "__main__":
    # 1. Load config
    config = c.config(experiment='amass_full_train_1', dataset='amass_dataset')

    # 2. Read the best model path from best_model.txt
    best_model_txt_path = os.path.join(config.checkpoint_path, "best_model.txt")
    with open(best_model_txt_path, "r") as f:
        lines = f.readlines()
    best_model_path = lines[0].strip()

    # 3. Load the best model  
    model = imu_uwb_pose_model.load_from_checkpoint(best_model_path, map_location=config.device, config=config)

    # 4. Instantiate the data module
    datamodule = imu_uwb_data_module(config)

    # 5. Create a trainer (configure GPU or CPU as needed)
    if config.device.type == 'cuda':
        accelerator = "gpu"
        devices = [0]
    else:
        accelerator = "cpu"
        devices = 1

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        fast_dev_run=True,
    )

    # 6. Run test using the loaded model
    output = trainer.predict(model, datamodule=datamodule)

    # run eval metrics on each output and optionally visualize
    # save true and predicted values for first output
    true = output[0]["true"]
    pred = output[0]["pred"]
    lengths = output[0]["lengths"]

    # save the true and predicted values
    torch.save(true, os.path.join('.', "true.pt"))
    torch.save(pred, os.path.join('.', "pred.pt"))
    torch.save(lengths, os.path.join('.', "lengths.pt"))