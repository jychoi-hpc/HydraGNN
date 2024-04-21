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
    parser.add_argument(
        "--nchannels",
        type=int,
        default=1,
        metavar="N",
        help="number of stream channels (default: 1)",
    )
    parser.add_argument("--log", help="log")
    parser.add_argument("--nsamples", type=int, help="number of samples")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--default", action="store_const", help="use default", dest="mode", const="default")
    group.add_argument("--stream", action="store_const", help="use stream", dest="mode", const="stream")
    group.add_argument("--shmemq", action="store_const", help="use shmemq", dest="mode", const="shmemq")
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
    os.environ["HYDRAGNN_USE_DDSTORE_EPOCH"] = "1"

    use_mq = 1 if args.mq else 0  ## 0: false, 1: true
    role = 1 if args.role == "consumer" else 0  ## 0: producer, 1: consumer
    mode = 1 if args.mode == "stream" else 0  ## 0: mq, 1: stream mq, 2: shmem mq
    mode = 2 if args.mode == "shmemq" else mode
    opt = {
        "preload": False,
        "shmem": False,
        "ddstore": True,
        "use_mq": use_mq,
        "role": role,
        "mode": mode,
    }
    # basedir = os.path.join(os.path.dirname(__file__), "dataset", "pickle")
    # trainset = SimplePickleDataset(basedir, "trainset")
    # valset = SimplePickleDataset(basedir, "valset")
    # testset = SimplePickleDataset(basedir, "testset")
    # trainset = DistDataset(trainset, "trainset", comm, **opt)
    fname = os.path.join(os.path.dirname(__file__), "dataset", "csce_gap.bp")
    trainset = AdiosDataset(fname, "trainset", comm, **opt)
    valset = AdiosDataset(fname, "valset", comm)
    testset = AdiosDataset(fname, "testset", comm)

    print(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    trainset_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=False
    )

    trainset_sample_list = list()
    for i in trainset_sampler:
        trainset_sample_list.append(i)
    if args.nsamples is not None:
        trainset_sample_list = trainset_sample_list[:args.nsamples]
    print(
        "local size: %d %d %d"
        % (
            len(trainset_sample_list),
            len(valset),
            len(testset),
        )
    )

    comm.Barrier()

    ## Consumer
    if role == 1:
        for k in range(args.epochs):
            comm.Barrier()
            t = 0
            for name, dataset, sample_list in zip(
                [
                    "trainset",
                ],
                [
                    trainset,
                ],
                [
                    trainset_sample_list,
                ],
            ):
                for seq, i in enumerate(sample_list):
                    if mode == 1:
                        i = 0
                    print(
                        ">>> [%d] consumer asking ... %s %d %d" % (rank, name, seq, i)
                    )
                    t0 = time.time()
                    dataset.__getitem__(i)
                    t1 = time.time()
                    print(
                        ">>> [%d] consumer received: %s %d %d (time: %f)"
                        % (rank, name, seq, i, t1 - t0)
                    )
                    t += t1 - t0
                print("[%d] consumer done. (avg: %f)" % (rank, t / len(dataset)))
                # comm.Barrier()
    ## Producer
    else:
        for k in range(args.epochs):
            comm.Barrier()
            t = 0
            for name, dataset, sample_list in zip(
                [
                    "trainset",
                ],
                [
                    trainset,
                ],
                [
                    trainset_sample_list,
                ],
            ):
                dataset.ddstore.epoch_begin()
                for seq, i in enumerate(sample_list):
                    if mode != 1:
                        i = 0
                        print(">>> [%d] producer waiting ..." % (rank))
                    else:
                        print(
                            ">>> [%d] producer streaming begin ... %s %d"
                            % (rank, name, i)
                        )
                    nchannels = args.nchannels if args.nchannels > 0 else 1
                    stream_ichannel = (seq // args.batch_size) % nchannels
                    t0 = time.time()
                    rtn = dataset.get(i, stream_ichannel=stream_ichannel)
                    t1 = time.time()
                    if mode != 1:
                        print(
                            ">>> [%d] producer responded:" % (rank),
                            name,
                            seq,
                            i,
                            stream_ichannel,
                            "(time: %f)" % (t1 - t0),
                        )
                    else:
                        print(
                            ">>> [%d] producer streaming end:" % (rank),
                            name,
                            seq,
                            i,
                            stream_ichannel,
                            "(time: %f)" % (t1 - t0),
                        )
                    t += t1 - t0
                    if (seq + 1) % args.batch_size == 0:
                        dataset.ddstore.epoch_end()
                        dataset.ddstore.epoch_begin()
                dataset.ddstore.epoch_end()
            print("[%d] producer done. (avg: %f)" % (rank, t / len(dataset)))
    sys.exit(0)
