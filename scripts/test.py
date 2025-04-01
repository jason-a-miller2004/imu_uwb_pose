import os
import pytorch_lightning as pl

from imu_uwb_pose import config as c
from imu_uwb_pose.training.imu_uwb_model import imu_uwb_pose_model
from imu_uwb_pose.training.utils import imu_uwb_data_module as imu_uwb_data_module


if __name__ == "__main__":
    # 1. Load config
    config = c.config(experiment='imu_uwb_train', dataset='amass_dataset')

    # 2. Read the best model path from best_model.txt
    best_model_txt_path = os.path.join(config.checkpoint_path, "best_model.txt")
    with open(best_model_txt_path, "r") as f:
        lines = f.readlines()
    best_model_path = lines[0].strip()

    # 3. Load the best model
    model = imu_uwb_pose_model.load_from_checkpoint(best_model_path)

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
        devices=devices
    )

    # 6. Run test using the loaded model
    trainer.test(model, datamodule=datamodule)
