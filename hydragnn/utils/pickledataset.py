import os
import glob
import pickle

import torch
from mpi4py import MPI

from .print_utils import print_distributed, log, iterate_tqdm

from hydragnn.utils.basedataset import BaseDataset
from hydragnn.utils.rawdataset import RawDataset


class SimplePickleDataset(BaseDataset):
    """Simple Pickle Dataset"""

    def __init__(self, basedir, label, subset=None):
        """
        Parameters
        ----------
        basedir: basedir
        label: label
        subset: a list of index to subset
        """
        self.basedir = basedir
        self.label = label
        self.subset = subset

        fname = "%s/%s-meta.pk" % (basedir, label)
        with open(fname, "rb") as f:
            self.minmax_node_feature = pickle.load(f)
            self.minmax_graph_feature = pickle.load(f)
            self.ntotal = pickle.load(f)

        log("Pickle files:", self.label, self.ntotal)

        if self.subset is None:
            self.subset = list(range(self.ntotal))

    def len(self):
        return len(self.subset)

    def get(self, i):
        k = self.subset[i]
        fname = "%s/%s-%d.pk" % (self.basedir, self.label, k)
        with open(fname, "rb") as f:
            data_object = pickle.load(f)
        return data_object

    def setsubset(self, subset):
        self.subset = subset


class SimplePickleWriter:
    """SimplePickleWriter class to write Torch Geometric graph data"""

    def __init__(
        self,
        dataset,
        basedir,
        label="total",
        minmax_node_feature=None,
        minmax_graph_feature=None,
        comm=MPI.COMM_WORLD,
    ):
        """
        Parameters
        ----------
        dataset: locally owned dataset (should be iterable)
        basedir: basedir
        label: label
        minmax_node_feature: minmax_node_feature
        minmax_graph_feature: minmax_graph_feature
        comm: MPI communicator
        """

        self.dataset = dataset
        if not isinstance(dataset, list):
            raise Exception("Unsuppored data type yet.")

        self.basedir = basedir
        self.label = label
        self.comm = comm
        self.rank = comm.Get_rank()

        self.minmax_node_feature = minmax_node_feature
        self.minmax_graph_feature = minmax_graph_feature

        ns = self.comm.allgather(len(self.dataset))
        noffset = sum(ns[: self.rank])
        ntotal = sum(ns)

        if self.rank == 0:
            if not os.path.exists(basedir):
                os.makedirs(basedir)
            fname = "%s/%s-meta.pk" % (basedir, label)
            with open(fname, "wb") as f:
                pickle.dump(self.minmax_node_feature, f)
                pickle.dump(self.minmax_graph_feature, f)
                pickle.dump(ntotal, f)
        comm.Barrier()

        for i, data in enumerate(self.dataset):
            fname = "%s/%s-%d.pk" % (basedir, label, noffset + i)
            with open(fname, "wb") as f:
                pickle.dump(data, f)
