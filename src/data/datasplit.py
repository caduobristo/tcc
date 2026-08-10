import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import default_collate


def custom_collate_fn(batch):
    """Filters out any None items before collating batch tensors."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


class DataSplit:
    """
    Splits dataset into reproducible train (80%), validation (10%), and test (10%) subsets.
    """

    def __init__(
        self,
        dataset,
        test_train_split: float = 0.8,
        val_train_split: float = 0.1,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        dataset_size = len(dataset)
        self.indices = list(range(dataset_size))

        if seed is not None:
            np.random.seed(seed)

        if shuffle:
            np.random.shuffle(self.indices)

        test_split_idx = int(np.floor(test_train_split * dataset_size))
        train_indices, self.test_indices = (
            self.indices[:test_split_idx],
            self.indices[test_split_idx:],
        )

        train_size = len(train_indices)
        val_split_idx = int(np.floor((1.0 - val_train_split) * train_size))
        self.train_indices, self.val_indices = (
            train_indices[:val_split_idx],
            train_indices[val_split_idx:],
        )

        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.val_sampler = SubsetRandomSampler(self.val_indices)
        self.test_sampler = SubsetRandomSampler(self.test_indices)

    def get_split(self, batch_size: int = 100, num_workers: int = 4, pin_memory: bool = False):
        """Returns train, validation, and test DataLoaders."""
        train_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=self.train_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=custom_collate_fn,
        )
        val_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=self.val_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=custom_collate_fn,
        )
        test_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=self.test_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=custom_collate_fn,
        )
        return train_loader, val_loader, test_loader
