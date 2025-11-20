import os
import tempfile
import torch
from torch.utils.data import Dataset

class footposer_dataset(Dataset):
    def __init__(self, config, train=True):
        self.train = train
        self.config = config
        self.root_dir = os.path.join(config.processed_pose, "FootPoser")
        cache_scope = "train" if train else "eval"
        self.cache_dir = os.path.join(config.cache_dir,tempfile.gettempdir(), "imu_uwb_pose_cache", "footposer", cache_scope)
        os.makedirs(self.cache_dir, exist_ok=True)
        self._length_cache = {}
        self.samples = self._build_index(config)
        

    def __len__(self):
        return len(self.samples)
    
    def _build_index(self, config):
        index = []

        if not os.path.exists(self.root_dir):
            return index

        subjects = os.listdir(self.root_dir)
        for subject in subjects:
            if (not self.train and subject != config.name and config.name != 'all'):
                continue
            if (self.train and (config.name == 'all' or config.name == subject) ):
                continue

            subject_dir = os.path.join(self.root_dir, subject)

            if not os.path.exists(subject_dir):
                continue

            actions = os.listdir(subject_dir)

            for action in actions:
                action_path = os.path.join(subject_dir, action)
                cached_path, seq_len = self._ensure_cached(action_path)
                for start in range(0, seq_len, config.max_sample_length):
                    end = min(start + config.max_sample_length, seq_len)
                    index.append((cached_path, start, end))
        return index

    def _cache_path(self, source_path):
        rel_path = os.path.relpath(source_path, self.root_dir)
        cache_path = os.path.join(self.cache_dir, rel_path)
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        return cache_path

    def _ensure_cached(self, source_path):
        cache_path = self._cache_path(source_path)
        if cache_path in self._length_cache:
            return cache_path, self._length_cache[cache_path]

        if os.path.exists(cache_path):
            data = torch.load(cache_path, map_location='cpu', weights_only=True)
            seq_len = data['x'].shape[0]
            self._length_cache[cache_path] = seq_len
            return cache_path, seq_len

        data = torch.load(source_path, map_location='cpu', weights_only=True)
        tmp_path = f"{cache_path}.tmp"
        torch.save(data, tmp_path)
        os.replace(tmp_path, cache_path)
        seq_len = data['x'].shape[0]
        self._length_cache[cache_path] = seq_len
        return cache_path, seq_len

    def _load_file(self, path):
        return torch.load(path, map_location='cpu', weights_only=True)

    def __getitem__(self, idx):
        path, start, end = self.samples[idx]
        data = self._load_file(path)
        return (data['x'][start:end], data['y'][start:end], data['joints'][start:end])
