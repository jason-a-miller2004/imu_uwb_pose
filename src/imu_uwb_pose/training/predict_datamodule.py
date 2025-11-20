from typing import Any, Optional, Sequence
from imu_uwb_pose.training.utils import pad_seq
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class SimplePredictDataset(Dataset):
    """Minimal dataset that exposes parallel x/y inputs for prediction."""

    def __init__(self, x: Sequence[Any], y: Sequence[Any], joints: Sequence[Any]) -> None:
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        self._x = x
        self._y = y
        self._joints = joints

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        return self._x[idx], self._y[idx], self._joints[idx]


class SimplePredictDataModule(pl.LightningDataModule):
    """DataModule that only supports prediction loading from provided x/y sequences."""

    def __init__(
        self,
        x: Sequence[Any],
        y: Sequence[Any],
        joints: Sequence[Any],
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        self._x = x
        self._y = y
        self._joints = joints

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._dataset: Optional[SimplePredictDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "predict"):
            self._dataset = SimplePredictDataset(self._x, self._y,self._joints)

    def predict_dataloader(self) -> DataLoader:
        if self._dataset is None:
            self.setup(stage="predict")
        return DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            collate_fn=pad_seq,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
        )
