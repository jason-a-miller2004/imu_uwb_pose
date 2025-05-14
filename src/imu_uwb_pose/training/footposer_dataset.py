import torch
from torch.utils.data import Dataset
import os

class footposer_dataset(Dataset):
    def __init__(self, config, train=True):
        self.train = train
        self.data = self.load_data(config)
        self.config = config
        

    def __len__(self):
        return len(self.x)
    
    def load_data(self, config):
        x = []
        y = []
        joints = []

        dir = os.path.join(config.processed_pose, "FootPoser", "train" if self.train else "test")

        subjects = os.listdir(dir)

        for subject in subjects:
            subject_dir = os.path.join(dir, subject)

            actions = os.listdir(subject_dir)

            for action in actions:
                action_path = os.path.join(subject_dir, action)
                data = torch.load(action_path, weights_only=True)

                x_split = torch.split(data['x'], config.max_sample_length)
                y_split = torch.split(data['y'], config.max_sample_length)
                joint_split = torch.split(data['joints'], config.max_sample_length)
                x.extend(x_split)
                y.extend(y_split)
                joints.extend(joint_split)
        self.x = x
        self.y = y
        self.joints = joints

    def __getitem__(self, idx):
        # Extract the angles
        return (self.x[idx], self.y[idx], self.joints[idx])