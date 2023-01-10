from abc import ABC, abstractmethod

import torch


class BaseDataset(torch.utils.data.Dataset, ABC):
    def __init__(self):
        super().__init__()
        self.dataset = list()

    @abstractmethod
    def get(self, idx):
        """
        Return a dataset at idx
        """
        pass

    @abstractmethod
    def len(self):
        """
        Total number of dataset.
        If data is distributed, it should be the global total size.
        """
        pass

    def apply(self, transform, inplace=False):
        dataset = []
        for data in self.dataset:
            dataset.append(transform(data))

        if not inplace:
            self.dataset = dataset

    def map(self, func):
        for data in self.dataset:
            yield func(data)

    def __len__(self):
        return self.len()

    def __getitem__(self, idx):
        return self.get(idx)

    def __iter__(self):
        for idx in range(self.len()):
            yield self.get(idx)
