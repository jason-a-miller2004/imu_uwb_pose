import torch
from torch.utils.data import Dataset
import os

class amass_dataset(Dataset):
    def __init__(self, config, train=True):
        self.train = train
        self.config = config
        self.samples = self._build_index(config)
        self._cache = {}
        
    def __len__(self):
        return len(self.samples)
    
    def _build_index(self, config):
        index = []
        dir_path = os.path.join(config.processed_pose, "AMASS", "train" if self.train else "test")

        if not os.path.exists(dir_path):
            return index

        for dataset_name in config.amass_datasets:
            dataset_dir = os.path.join(dir_path, dataset_name)

            if not os.path.exists(dataset_dir):
                continue

            subjects = os.listdir(dataset_dir)

            for subject in subjects:
                subject_dir = os.path.join(dataset_dir, subject)

                if not os.path.exists(subject_dir):
                    continue

                actions = os.listdir(subject_dir)

                for action in actions:
                    action_path = os.path.join(subject_dir, action)
                    data = torch.load(action_path, map_location='cpu', weights_only=True)
                    seq_len = data['x'].shape[0]

                    for start in range(0, seq_len, config.max_sample_length):
                        end = min(start + config.max_sample_length, seq_len)
                        index.append((action_path, start, end))
                    del data
        return index

    def _load_file(self, path):
        if path not in self._cache:
            data = torch.load(path, map_location='cpu', weights_only=True)
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    data[key] = value.share_memory_()
            self._cache[path] = data
        return self._cache[path]

    def __getitem__(self, idx):
        path, start, end = self.samples[idx]
        data = self._load_file(path)
        return (data['x'][start:end], data['trans'][start:end], data['joints'][start:end])
