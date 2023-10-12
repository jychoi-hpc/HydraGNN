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
        "use_mq": False,
        "role": role,
    }

    fname = args.input
    trainset = SimplePickleDataset(fname, "trainset")
    trainset = DistDataset(trainset, "trainset", **opt)
    print("trainset size: %d" % len(trainset))

    comm = MPI.COMM_WORLD
    comm.Barrier()

    if role == 1:
        for i in range(len(trainset)):
            print("get:", i)
            trainset.get(i)
        comm.Barrier()
        #trainset.get(-1)
    else:
        while True:
            rtn = trainset.get(0)
            print(rtn)
            if rtn == -1:
                break

    sys.exit(0)
