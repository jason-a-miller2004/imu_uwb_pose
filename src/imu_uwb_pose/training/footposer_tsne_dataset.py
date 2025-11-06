import torch
from torch.utils.data import Dataset
import os
import re

class footposer_tsne_dataset(Dataset):
    def __init__(self, config, train=True):
        self.train = train
        self.data = self.load_data(config)
        self.config = config
        

    def __len__(self):
        return len(self.x)
    
    def load_data(self, config):
        x = []
        subject_uid = []
        motion_uid = []

        dir = os.path.join(config.processed_pose, "FootPoser")

        if not os.path.exists(dir):
            self.x = x
            self.subject_uid = subject_uid
            self.motion_uid = motion_uid
            return

        subjects = os.listdir(dir)
        for subject in subjects:
            subject_dir = os.path.join(dir, subject)
            if not os.path.exists(subject_dir):
                continue

            actions = os.listdir(subject_dir)

            for action in actions:

                action_path = os.path.join(subject_dir, action)
                data = torch.load(action_path, weights_only=True)

                x_split = torch.split(data['x'], config.max_sample_length)
                x.extend(x_split)

                # Extract motion name (e.g., 'activities' from 'activities1.pt')
                m = re.match(r'^([A-Za-z_]+)(\d+)\.pt$', action)
                if not m:
                    raise ValueError(f"Unexpected action filename format: {action}")
                motion_name = m.group(1)

                motion_uid.extend([config.motion_uids[motion_name]] * len(x_split))
                subject_uid.extend([config.participant_uids[subject]] * len(x_split))
        self.x = x
        self.subject_uid = subject_uid
        self.motion_uid = motion_uid

    def __getitem__(self, idx):
        # Extract the angles
        return (self.x[idx], self.subject_uid[idx], self.motion_uid[idx])