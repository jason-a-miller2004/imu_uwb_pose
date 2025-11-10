import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from imu_uwb_pose.training.RNN import RNN

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

def pad_seq_tsne(batch):
    xs, subj_ids, motion_ids = zip(*batch)           # tuples
    lens = [x.shape[0] for x in xs]                  # time lengths
    xs = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    return xs, lens, list(subj_ids), list(motion_ids)

def get_dataset(config):
    dataset = config.dataset
    # load the dataset
    if dataset == "amass_dataset":
        from imu_uwb_pose.training import amass_dataset as dataset
        train_dataset = dataset.amass_dataset(config)
        test_dataset = dataset.amass_dataset(config, train=False)
    elif dataset == "footposer_dataset":
        from imu_uwb_pose.training import footposer_dataset as dataset
        train_dataset = dataset.footposer_dataset(config)
        test_dataset = dataset.footposer_dataset(config, train=False)
    else:
        print("Enter a valid model")
        return

    # get the train and val split
    train_size, val_size = train_val_split(train_dataset, train_pct=config.train_pct)

    # split the dataset
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    return train_dataset, test_dataset, val_dataset

def get_model(config, model_name):
    n_input = 6 * len(config.absolute_joint_angles) + len(config.uwb_dists) + len(config.uwb_floor_dists)
    n_output = 132
    match model_name:
        case 'bilstm_one_layer':
            return RNN(n_rnn_layer=1, n_input=n_input, n_output=n_output, n_hidden=512, bidirectional=True)
        case 'bilstm_two_layer':
            return RNN(n_rnn_layer=2, n_input=n_input, n_output=n_output, n_hidden=512, bidirectional=True)
        case 'bilstm_three_layer':
            return RNN(n_rnn_layer=3, n_input=n_input, n_output=n_output, n_hidden=512, bidirectional=True)
        case 'bilstm_four_layer':
            return RNN(n_rnn_layer=4, n_input=n_input, n_output=n_output, n_hidden=512, bidirectional=True)
        case _:
            raise ValueError("Not a valid model name")

class imu_uwb_data_module(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        self.train_dataset, self.test_dataset, self.val_dataset = get_dataset(self.config)
        print("Done with setup")

    def _loader_kwargs(self, *, shuffle):
        num_workers = max(0, self.config.num_workers)
        kwargs = dict(
            batch_size=self.config.batch_size,
            collate_fn=pad_seq,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False,
            prefetch_factor=2
        )
        if num_workers > 0:
            kwargs["persistent_workers"] = self.config.persistent_workers
        return kwargs

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self._loader_kwargs(shuffle=True))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self._loader_kwargs(shuffle=False))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self._loader_kwargs(shuffle=False))
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, **self._loader_kwargs(shuffle=False))
