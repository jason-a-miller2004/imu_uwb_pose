import torch
from torch.utils.data import Dataset
import os

class footposer_dataset(Dataset):
    def __init__(self, config, train=True):
        self.train = train
        self.config = config
        self.samples = self._build_index(config)
        self._cache = {}
        

    def __len__(self):
        return len(self.samples)
    
    def _build_index(self, config):
        index = []
        dir_path = os.path.join(config.processed_pose, "FootPoser")

        if not os.path.exists(dir_path):
            return index

        subjects = os.listdir(dir_path)
        for subject in subjects:
            if (not self.train and subject != config.name and config.name != 'all'):
                continue
            if (self.train and (config.name == 'all' or config.name == subject) ):
                continue

            subject_dir = os.path.join(dir_path, subject)

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
        return (data['x'][start:end], data['y'][start:end], data['joints'][start:end])
