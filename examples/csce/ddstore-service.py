import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import sys
import argparse
from tqdm import tqdm
from mpi4py import MPI

import torch
from torchvision import datasets, transforms

import time
import numpy as np

import torch.distributed as dist
import os
import socket
import psutil
import re

import hydragnn
from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
from hydragnn.utils.distdataset import DistDataset
from hydragnn.utils.pickledataset import SimplePickleWriter, SimplePickleDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--mq", action="store_true", help="use mq")
    parser.add_argument("--stream", action="store_true", help="use stream mode")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--producer",
        help="producer",
        action="store_const",
        dest="role",
        const="producer",
    )
    group.add_argument(
        "--consumer",
        help="consumer",
        action="store_const",
        dest="role",
        const="consumer",
    )
    parser.set_defaults(role="consumer")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    comm_size, rank = hydragnn.utils.setup_ddp()
    device = torch.device("cpu")
    print("DDP setup:", comm_size, rank, device)

    os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
    os.environ["HYDRAGNN_USE_ddstore"] = "1"

    use_mq = 1 if args.mq else 0  ## 0: false, 1: true
    role = 1 if args.role == "consumer" else 0  ## 0: producer, 1: consumer
    mode = 1 if args.stream else 0  ## 0: mq, 1: stream mq
    opt = {
        "use_mq": use_mq,
        "role": role,
        "mode": mode,
    }
    basedir = os.path.join(os.path.dirname(__file__), "dataset", "pickle")
    trainset = SimplePickleDataset(basedir, "trainset")
    valset = SimplePickleDataset(basedir, "valset")
    testset = SimplePickleDataset(basedir, "testset")
    trainset = DistDataset(trainset, "trainset", comm, **opt)
    valset = DistDataset(valset, "valset", comm, **opt)
    testset = DistDataset(testset, "testset", comm, **opt)
    print(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    trainset_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    valset_sampler = torch.utils.data.distributed.DistributedSampler(valset)
    testset_sampler = torch.utils.data.distributed.DistributedSampler(testset)

    trainset_sample_list = list()
    valset_sample_list = list()
    testset_sample_list = list()
    for i in trainset_sampler:
        trainset_sample_list.append(i)
    for i in valset_sampler:
        valset_sample_list.append(i)
    for i in testset_sampler:
        testset_sample_list.append(i)
    print(
        "local size: %d %d %d"
        % (
            len(trainset_sample_list),
            len(trainset_sample_list),
            len(testset_sample_list),
        )
    )

    comm.Barrier()

    if role == 1:
        for k in range(args.epochs):
            comm.Barrier()
            t = 0
            for name, dataset, sample_list in zip(
                ["trainset", "valset", "testset"],
                [trainset, valset, testset],
                [trainset_sample_list, valset_sample_list, testset_sample_list],
            ):
                for i in sample_list:
                    if mode == 1:
                        i = 0
                    print(">>> [%d] consumer asking ... %s %d" % (rank, name, i))
                    t0 = time.time()
                    dataset.__getitem__(i)
                    t1 = time.time()
                    print(
                        ">>> [%d] consumer received: %s %d (time: %f)"
                        % (rank, name, i, t1 - t0)
                    )
                    t += t1 - t0
                print("[%d] consumer done. (avg: %f)" % (rank, t / len(dataset)))
                # comm.Barrier()
    else:
        for k in range(args.epochs):
            comm.Barrier()
            for name, dataset, sample_list in zip(
                ["trainset", "valset", "testset"],
                [trainset, valset, testset],
                [trainset_sample_list, valset_sample_list, testset_sample_list],
            ):
                dataset.ddstore.epoch_begin()
                for seq, i in enumerate(sample_list):
                    if mode == 0:
                        i = 0
                        print(">>> [%d] producer waiting ..." % (rank))
                    else:
                        print(
                            ">>> [%d] producer streaming begin ... %s %d"
                            % (rank, name, i)
                        )
                    rtn = dataset.get(i)
                    if mode == 0:
                        print(">>> [%d] producer responded: %s %d" % (rank, name, i))
                    else:
                        print(
                            ">>> [%d] producer streaming end: %s %d" % (rank, name, i)
                        )
                    if (seq + 1) % args.batch_size == 0:
                        dataset.ddstore.epoch_end()
                        dataset.ddstore.epoch_begin()
                dataset.ddstore.epoch_end()
    sys.exit(0)
