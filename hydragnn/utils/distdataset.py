from mpi4py import MPI
import numpy as np

import torch
import torch_geometric.data

from hydragnn.utils.abstractbasedataset import AbstractBaseDataset

try:
    import pyddstore2 as dds
except ImportError:
    pass

from hydragnn.utils.print_utils import log
from hydragnn.utils import nsplit

import hydragnn.utils.tracer as tr
import pickle


class DistDataset(AbstractBaseDataset):
    """Distributed dataset class"""

    def __init__(
        self,
        data,
        label,
        comm=MPI.COMM_WORLD,
        ddstore_width=None,
        use_mq=False,
        role=1,
        mode=0,
    ):
        super().__init__()

        self.label = label
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()
        self.ddstore_width = (
            ddstore_width if ddstore_width is not None else self.comm_size
        )
        self.ddstore_comm = self.comm.Split(self.rank // self.ddstore_width, self.rank)
        self.ddstore_comm_rank = self.ddstore_comm.Get_rank()
        self.ddstore_comm_size = self.ddstore_comm.Get_size()
        self.ddstore = dds.PyDDStore(
            self.ddstore_comm, use_mq=use_mq, role=role, mode=mode
        )

        ## set total before set subset
        self.total_ns = len(data)
        print("init: total_ns =", self.total_ns)

        rx = list(nsplit(range(len(data)), self.ddstore_comm_size))[
            self.ddstore_comm_rank
        ]
        for i in rx:
            self.dataset.append(data[i])
        print(self.rank, len(self.dataset))

        self.data = list()
        self.labels = list()

        self.use_mq = use_mq
        self.role = role
        self.ddstore.add(label, self.dataset)

    def len(self):
        return self.total_ns

    def __len__(self):
        return self.len()

    @tr.profile("get")
    def get(self, idx):
        data_object = self.ddstore.get(
            self.label, idx, decoder=lambda x: pickle.loads(x)
        )
        return data_object

    def __getitem__(self, idx):
        return self.get(idx)

    def __del__(self):
        if self.ddstore:
            self.ddstore.free()
