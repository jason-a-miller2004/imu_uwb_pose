import os
import tempfile
import torch
from torch.utils.data import Dataset
from imu_uwb_pose.utils import rotation_matrix_to_r6d

class amass_dataset(Dataset):
    def __init__(self, config, train=True):
        self.train = train
        self.config = config
        split = "train" if self.train else "test"
        self.root_dir = os.path.join(config.processed_pose, "AMASS", split)
        cache_scope = "train" if self.train else "test"
        self.cache_dir = os.path.join(config.cache_dir, tempfile.gettempdir(), "imu_uwb_pose_cache", "amass", cache_scope)
        os.makedirs(self.cache_dir, exist_ok=True)
        self._length_cache = {}
        self.samples = self._build_index(config)
        
    def __len__(self):
        return len(self.samples)
    
    def _build_index(self, config):
        index = []

        if not os.path.exists(self.root_dir):
            return index

        for dataset_name in config.amass_datasets:
            dataset_dir = os.path.join(self.root_dir, dataset_name)

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
            if self._needs_conversion(data['x']):
                data['x'] = self._convert_legacy_input(data['x'])
                self._write_cache(cache_path, data)
            seq_len = data['x'].shape[0]
            self._length_cache[cache_path] = seq_len
            return cache_path, seq_len

        data = torch.load(source_path, map_location='cpu', weights_only=True)
        data['x'] = self._convert_legacy_input(data['x'])
        self._write_cache(cache_path, data)
        seq_len = data['x'].shape[0]
        self._length_cache[cache_path] = seq_len
        return cache_path, seq_len

    def _write_cache(self, cache_path, data):
        tmp_path = f"{cache_path}.tmp"
        torch.save(data, tmp_path)
        os.replace(tmp_path, cache_path)

    def _load_file(self, path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if self._needs_conversion(data['x']):
            data['x'] = self._convert_legacy_input(data['x'])
            self._write_cache(path, data)
        return data

    def _needs_conversion(self, x):
        return isinstance(x, torch.Tensor) and x.shape[-1] == 27

    def _convert_legacy_input(self, x):
        if not isinstance(x, torch.Tensor):
            return x

        if x.shape[-1] != 27:
            return x

        batch_shape = x.shape[:-1]
        left_mat = x[..., :9].reshape(-1, 3, 3)
        right_mat = x[..., 9:18].reshape(-1, 3, 3)

        left_r6d = rotation_matrix_to_r6d(left_mat).reshape(*batch_shape, 6)
        right_r6d = rotation_matrix_to_r6d(right_mat).reshape(*batch_shape, 6)

        rest = x[..., 18:21]
        return torch.cat([left_r6d, right_r6d, rest], dim=-1)

    def __getitem__(self, idx):
        path, start, end = self.samples[idx]
        data = self._load_file(path)
        return (data['x'][start:end], data['y'][start:end], data['joints'][start:end])
