import mpi4py

mpi4py.rc.thread_level = "serialized"
mpi4py.rc.threads = False

import os, json
import logging
import sys
from mpi4py import MPI
import argparse
import time

import hydragnn
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.config_utils import get_log_name_config
from hydragnn.preprocess.raw_dataset_loader import RawDataLoader
from hydragnn.utils.model import print_model
from hydragnn.utils.print_utils import iterate_tqdm, log
from hydragnn.utils.distdataset import DistDataset

import torch
import torch.distributed as dist


def flatten(l):
    return [item for sublist in l for item in sublist]


def check_retainable_connections(data, edge_index):
    # C-C remove, Si-Si remove, Si-O remove O-O remove.
    assert (
        edge_index < data.edge_index.shape[1]
    ), "Edge index exceeds total number of edges available"

    if (
        data.edge_index[0, edge_index] == 0 and data.edge_index[1, edge_index] != 0
    ) or (data.edge_index[1, edge_index] == 0 and data.edge_index[0, edge_index] != 0):
        return True
    else:
        return False


def remove_edges(data):
    ## (2023/1) jyc: Iterating with index is slow. Generating mask is faster
    # edges_to_retain = [ check_retainable_connections(data, index) for index in range(0,data.edge_index.shape[1]) ]
    idx = torch.where(data.x == 0.0)[0]  ## node index for "C"
    edges_to_retain = torch.any(data.edge_index == idx[0], dim=0)
    if len(idx) > 1:  ## in case if there are multiple "C"s
        for i in range(1, len(idx)):
            edges_to_retain = torch.logical_or(
                edges_to_retain, torch.any(data.edge_index == idx[i], dim=0)
            )
    data.edge_index = data.edge_index[:, edges_to_retain]
    data.edge_attr = data.edge_attr[edges_to_retain, :]
    return data


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loadexistingsplit",
        action="store_true",
        help="loading from existing pickle/adios files with train/test/validate splits",
    )
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only. Adios or pickle saving and no train",
    )
    parser.add_argument(
        "--distds",
        action="store_true",
        help="distds dataset",
    )
    parser.add_argument(
        "--inputfile", help="input file", type=str, default="zeolite.json"
    )
    parser.add_argument("--sampling", help="sampling ratio", type=float, default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adios",
        help="Adios dataset",
        action="store_const",
        dest="format",
        const="adios",
    )
    group.add_argument(
        "--pickle",
        help="Pickle dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )
    parser.set_defaults(format="pickle")

    args = parser.parse_args()

    dirpwd = os.path.dirname(__file__)
    input_filename = os.path.join(dirpwd, args.inputfile)
    with open(input_filename, "r") as f:
        config = json.load(f)
    hydragnn.utils.setup_log(get_log_name_config(config))
    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.setup_ddp()
    ##################################################################################################################
    comm = MPI.COMM_WORLD
    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    os.environ["SERIALIZED_DATA_PATH"] = dirpwd + "/dataset"
    datasetname = config["Dataset"]["name"]
    fname_adios = dirpwd + "/dataset/%s.bp" % (datasetname)
    config["Dataset"]["name"] = "%s_%d" % (datasetname, rank)
    if not args.loadexistingsplit:
        for dataset_type, raw_data_path in config["Dataset"]["path"].items():
            if not os.path.isabs(raw_data_path):
                raw_data_path = os.path.join(dirpwd, raw_data_path)
            if not os.path.exists(raw_data_path):
                raise ValueError("Folder not found: ", raw_data_path)
            config["Dataset"]["path"][dataset_type] = raw_data_path

        ## each process saves its own data file
        loader = RawDataLoader(config["Dataset"], dist=True)
        loader.load_raw_data(sampling=args.sampling)

        ## Read total pkl and split (no graph object conversion)
        hydragnn.preprocess.total_to_train_val_test_pkls(config, isdist=True)

        if args.format == "adios":
            ## Read each pkl and graph object conversion with max-edge normalization
            (
                trainset,
                valset,
                testset,
            ) = hydragnn.preprocess.load_data.load_train_val_test_sets(
                config, isdist=True
            )

            ## remove edges
            for dataset in [trainset, valset, testset]:
                for data in iterate_tqdm(
                    dataset, verbosity_level=2, desc="Remove edges"
                ):
                    remove_edges(data)

            from hydragnn.utils.adiosdataset import AdiosWriter

            adwriter = AdiosWriter(fname_adios, comm)
            adwriter.add("trainset", trainset)
            adwriter.add("valset", valset)
            adwriter.add("testset", testset)
            adwriter.add_global("minmax_node_feature", loader.minmax_node_feature)
            adwriter.add_global("minmax_graph_feature", loader.minmax_graph_feature)
            adwriter.save()
    if args.preonly:
        sys.exit(0)

    timer = Timer("load_data")
    timer.start()
    if args.format == "adios":
        from hydragnn.utils.adiosdataset import AdiosDataset

        info("Adios load")
        trainset = AdiosDataset(fname_adios, "trainset", comm, distds=args.distds)
        valset = AdiosDataset(fname_adios, "valset", comm)
        testset = AdiosDataset(fname_adios, "testset", comm)
        ## Set minmax read from bp file
        config["NeuralNetwork"]["Variables_of_interest"][
            "minmax_node_feature"
        ] = trainset.minmax_node_feature
        config["NeuralNetwork"]["Variables_of_interest"][
            "minmax_graph_feature"
        ] = trainset.minmax_graph_feature

        if args.distds:
            os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
            os.environ["HYDRAGNN_USE_DISTDS"] = "1"
    elif args.format == "pickle":
        config["Dataset"]["path"] = {}
        ##set directory to load processed pickle files, train/validate/test
        for dataset_type in ["train", "validate", "test"]:
            raw_data_path = f"{os.environ['SERIALIZED_DATA_PATH']}/serialized_dataset/{config['Dataset']['name']}_{dataset_type}.pkl"
            config["Dataset"]["path"][dataset_type] = raw_data_path

        info("Pickle load")
        (
            trainset,
            valset,
            testset,
        ) = hydragnn.preprocess.load_data.load_train_val_test_sets(config, isdist=True)

        ## Remove edes. It is better to call before gathering since it reduces the size.
        info("Remove edges")
        for dataset in [trainset, valset, testset]:
            for data in dataset:
                remove_edges(data)

        if args.distds:
            trainset = DistDataset(trainset, "trainset")
            valset = DistDataset(valset, "valset")
            testset = DistDataset(testset, "testset")
            os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
            os.environ["HYDRAGNN_USE_DISTDS"] = "1"
        else:
            # FIXME: Use MPI to gather data. This is no good either. It can be a problem if data size is larger than MPI capacity.
            info("Gather dataset")
            t0 = time.time()
            comm = MPI.COMM_WORLD
            trainset_all = comm.allgather(trainset)
            valset_all = comm.allgather(valset)
            testset_all = comm.allgather(testset)
            trainset = flatten(trainset_all)
            valset = flatten(valset_all)
            testset = flatten(testset_all)
            t1 = time.time()
            log("Time:", t1 - t0)
    else:
        raise ValueError("Unknown data format: %d" % args.format)

    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    info("create_dataloaders")
    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )
    timer.stop()

    info("update_config")
    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_node_feature", None)
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_graph_feature", None)

    info("create_model_config")
    verbosity = config["Verbosity"]["level"]
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    if rank == 0:
        print_model(model)
    comm.Barrier()

    model = hydragnn.utils.get_distributed_model(model, verbosity)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    log_name = get_log_name_config(config)
    writer = hydragnn.utils.get_summary_writer(log_name)

    if dist.is_initialized():
        dist.barrier()
    with open("./logs/" + log_name + "/config.json", "w") as f:
        json.dump(config, f)

    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
        create_plots=True,
    )

    hydragnn.utils.save_model(model, optimizer, log_name)
    hydragnn.utils.print_timers(verbosity)

    sys.exit(0)
