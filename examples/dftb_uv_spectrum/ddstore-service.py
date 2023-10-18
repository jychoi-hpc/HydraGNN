import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import sys
import argparse
from tqdm import tqdm
from mpi4py import MPI

from hydragnn.utils.adiosdataset import AdiosDataset
from hydragnn.utils.distdataset import DistDataset
from hydragnn.utils.pickledataset import SimplePickleDataset
import hydragnn.utils.tracer as tr
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input", help="input filename")
    parser.add_argument("--width", type=int, help="ddstore width", default=6)
    parser.add_argument("--mq", action="store_true", help="use mq")
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

    role = 1 if args.role == "consumer" else 0  ## 0: producer, 1: consumer
    opt = {
        "ddstore_width": args.width,
        "ddstore_version": 2,
        "use_mq": args.mq,
        "role": role,
    }

    fname = args.input
    trainset = SimplePickleDataset(fname, "trainset")
    trainset = DistDataset(trainset, "trainset", **opt)
    print("trainset size: %d" % len(trainset))

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm.Barrier()

    if role == 1:
        t = 0
        for i in range(len(trainset)):
            print(">>> [%d] consumer asking ... %d" % (rank, i))
            t0 = time.time()
            trainset.get(i)
            t1 = time.time()
            print(">>> [%d] consumer received: %d (time: %f)" % (rank, i, t1 - t0))
            t += t1 - t0
        print("[%d] consumer done. (avg: %f)" % (rank, t / len(trainset)))
        comm.Barrier()
    else:
        # trainset.ddstore.epoch_begin()
        cnt = 0
        while True:
            print(">>> [%d] producer waiting ..." % (rank))
            rtn = trainset.get(0)
            print(">>> [%d] producer responded." % (rank))
            cnt += 1
            # comm.Barrier()
            """
            if cnt%500:
                comm.Barrier()
                #trainset.ddstore.epoch_end()
                #trainset.ddstore.epoch_begin()
            """
        # trainset.ddstore.epoch_end()
    sys.exit(0)
