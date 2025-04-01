import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

def train_val_split(dataset, train_pct):
    # get the train and val split
    total_size = len(dataset)
    train_size = int(train_pct * total_size)
    val_size = total_size - train_size
    return train_size, val_size

def pad_seq(batch):
    inputs = [item[0] for item in batch]
    outputs = [item[1] for item in batch]
    poses = [item[2] for item in batch]
    
    input_lens = [item.shape[0] for item in inputs]
    output_lens = [item.shape[0] for item in outputs]
    
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    outputs = nn.utils.rnn.pad_sequence(outputs, batch_first=True)
    poses = nn.utils.rnn.pad_sequence(poses, batch_first=True)
    return inputs, outputs, poses, input_lens, output_lens

def get_dataset(config):
    dataset = config.dataset
    # load the dataset
    if dataset == "amass_dataset":
        from imu_uwb_pose.training import amass_dataset as dataset
        train_dataset = dataset.amass_dataset(config)
        test_dataset = dataset.amass_dataset(config, train=False)

    else:
        print("Enter a valid model")
        return

    # get the train and val split
    train_size, val_size = train_val_split(train_dataset, train_pct=config.train_pct)

    # split the dataset
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    return train_dataset, test_dataset, val_dataset

class imu_uwb_data_module(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        self.train_dataset, self.test_dataset, self.val_dataset = get_dataset(self.config)
        print("Done with setup")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, collate_fn=pad_seq, num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size, collate_fn=pad_seq, num_workers=8, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.batch_size, collate_fn=pad_seq, num_workers=8, shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.batch_size, collate_fn=pad_seq, num_workers=8, shuffle=False)