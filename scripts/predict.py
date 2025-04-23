import os
import argparse
import pytorch_lightning as pl
import torch
import numpy as np

from imu_uwb_pose import config as c
from imu_uwb_pose.training.imu_uwb_model import imu_uwb_pose_model
from imu_uwb_pose.training.utils import imu_uwb_data_module as imu_uwb_data_module
from imu_uwb_pose.utils import rotation_matrix_to_r6d, r6d_to_axis_angle
import torch.nn as nn
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict movement from sensors")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="file path of the model to use. Corresponds to config.experiment."
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="file path of the data to use"
    )

    args = parser.parse_args()

    config = c.config(
        experiment=args.model
    )

    best_model_txt_path = os.path.join(config.checkpoint_path, "best_model.txt")
    with open(best_model_txt_path, "r") as f:
        lines = f.readlines()
    best_model_path = lines[0].strip()

    print(f"Loading the best model from: {best_model_path}")

    model = imu_uwb_pose_model.load_from_checkpoint(
        best_model_path,
        map_location=config.device,
        config=config
    )

    print(f'Loading the data from drive')
    # load the data as tensors
    with open(args.data, 'rb') as file:
        data = pickle.load(file)

    # convert imu rotation matrix to r6d
    left_imu = rotation_matrix_to_r6d(torch.tensor(data['left_imu'], dtype=torch.float32))
    right_imu = rotation_matrix_to_r6d(torch.tensor(data['right_imu'], dtype=torch.float32))
    uwb_dists = data['uwb']

    # concat all the fields into left imu, right imu, uwb dists
    data = np.concatenate([left_imu, right_imu, uwb_dists], axis=1)

    # split data into config.max_sample_length chunks and batch size
    data_split = []
    data_split.extend(torch.split(torch.tensor(data, dtype=torch.float32), config.max_sample_length))
    print("len of data split " + str(len(data_split)))
    print("finished processing data to correct format")

    results = []
    idx = 0
    batch = 0
    while (idx < len(data_split)):
        # batch with size config.batch_size
        print("processing batch: " + str(batch))
        input = data_split[idx : min(idx + config.batch_size, len(data_split))]
        input_lens = [item.shape[0] for item in input]
        input = nn.utils.rnn.pad_sequence(input, batch_first=True)
        idx += config.batch_size
        batch += 1
        # send to device
        input = input.to(config.device)

        # make prediction
        with torch.no_grad():
            pred = model(input, input_lens)

        print(f"Prediction shape: {pred.shape}")
        batch_size = pred.shape[0]
        pred = pred.reshape(-1, 6)
        # convert prediction from r6d to axis angle
        pred = r6d_to_axis_angle(pred)
        pred = pred.reshape(batch_size, config.max_sample_length, -1, 3)

        # add to results
        results.append((pred, input_lens))

    # save results
    print("Saving results to: ./results.pt")
    torch.save(results, './results.pt')
