import os, sys

import math
import torch

torch.backends.cudnn.enabled = False

# FIX random seed
random_state = 0
torch.manual_seed(random_state)

# deprecated in torch_geometric 2.0
try:
    from torch_geometric.loader import DataLoader
except:
    from torch_geometric.data import DataLoader

import pandas as pd
import subprocess
import re
import argparse
import time
import glob
import threading

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.width = None

# Retrieve constants
NNODES = int(os.environ["NNODES"])
NTOTGPUS = int(os.environ["NTOTGPUS"])
NNODES_PER_TRIAL = int(os.environ["NNODES_PER_TRIAL"])
NGPUS_PER_TRIAL = int(os.environ["NGPUS_PER_TRIAL"])
NUM_CONCURRENT_TRIALS = int(os.environ["NUM_CONCURRENT_TRIALS"])
NTOT_DEEPHYPER_RANKS = int(os.environ["NTOT_DEEPHYPER_RANKS"])
OMP_NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])
GPUS_PER_NODE = int(os.getenv("GPUS_PER_NODE", "8"))
HYDRAGNN_SYSTEM = os.getenv("HYDRAGNN_SYSTEM", "frontier")
DEEPHYPER_LOG_DIR = os.environ["DEEPHYPER_LOG_DIR"]
SLURM_JOB_ID = os.environ["SLURM_JOB_ID"]
OMNISTAT_DIR = os.getenv("OMNISTAT_DIR", None)

# HydraGNN-specific environment variables
MULTI_MODEL_LIST = os.environ["MULTI_MODEL_LIST"]
NUM_EPOCH = int(os.environ["NUM_EPOCH"])
BATCH_SIZE = int(os.environ["BATCH_SIZE"])
HYDRAGNN_MAX_NUM_BATCH = int(os.environ["HYDRAGNN_MAX_NUM_BATCH"])
## DDStore method 2: trials read the data a core step holds on CORE_NODES
## (see ddstore-core.sh); unset for method 1
DDSTORE_HANDSHAKE_DIR = (
    os.getenv("HYDRAGNN_DDSTORE_HANDSHAKE_DIR")
    if os.getenv("HYDRAGNN_DDSTORE_METHOD") == "2"
    else None
)
DDSTORE_GROUP_MAP_FILE = os.getenv("DDSTORE_GROUP_MAP_FILE")
## Nodes kept for the DDStore core (ddstore-core.sh); also left out of the
## queue by a method-1 search in the same job (COMPARE_TIMINGS=1)
DDSTORE_CORE_NODES = [n for n in os.getenv("CORE_NODES", "").split(",") if n]
## Separates the searches of one job (COMPARE_TIMINGS=1)
RUN_TAG = os.getenv("HYDRAGNN_RUN_TAG")
VNI_WRAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vni-wrap.sh")


def report_trial_progress(stop_event, interval=5):
    """Summarize trial logs on the controller without streaming every rank's output."""
    offsets = {}
    active = set()
    last_heartbeat = time.monotonic()

    def scan_logs():
        for path in sorted(glob.glob(os.path.join(DEEPHYPER_LOG_DIR, "output-*.txt"))):
            trial_id = os.path.basename(path)[len("output-") : -len(".txt")]
            try:
                with open(path) as log:
                    log.seek(offsets.get(path, 0))
                    lines = log.readlines()
                    offsets[path] = log.tell()
            except OSError:
                continue
            for line in lines:
                if line.startswith("Command = "):
                    active.add(trial_id)
                    print(f"[DeepHyper] Trial {trial_id} started; logs: {DEEPHYPER_LOG_DIR}", flush=True)
                elif line.startswith("Output:"):
                    active.discard(trial_id)
                    print(f"[DeepHyper] Trial {trial_id} finished: {line.strip()}", flush=True)

            error_path = os.path.join(DEEPHYPER_LOG_DIR, f"error_{SLURM_JOB_ID}_{trial_id}.txt")
            if os.path.exists(error_path):
                try:
                    with open(error_path) as log:
                        log.seek(offsets.get(error_path, 0))
                        lines = log.readlines()
                        offsets[error_path] = log.tell()
                except OSError:
                    continue
                for line in lines:
                    if "Epoch:" in line or "Data loading time (sec):" in line:
                        print(f"[DeepHyper] Trial {trial_id}: {line.strip()}", flush=True)

    while True:
        scan_logs()
        if active and time.monotonic() - last_heartbeat >= 30:
            print(f"[DeepHyper] Waiting on trial(s): {', '.join(sorted(active))}", flush=True)
            last_heartbeat = time.monotonic()
        if stop_event.wait(interval):
            break
    scan_logs()


def to_float(x):
    x = x.lower()
    if x == "nan":
        return math.nan
    if x in ("inf", "+inf"):
        return math.inf
    if x == "-inf":
        return -math.inf
    return float(x)


def run(trial, dequed=None):
    os.makedirs(f"{DEEPHYPER_LOG_DIR}", exist_ok=True)
    f = open(f"{DEEPHYPER_LOG_DIR}/output-{trial.id}.txt", "w")
    python_exe = sys.executable
    python_script = os.path.join(os.path.dirname(__file__), "gfm_mlip_all_mpnn.py")

    log_name = f"gfm_{SLURM_JOB_ID}_{trial.id}"
    if RUN_TAG is not None:
        log_name = f"gfm_{SLURM_JOB_ID}_{RUN_TAG}_{trial.id}"
    # In grouped method 2, the queue holds whole node groups as one resource.
    trial_nodes = dequed[0] if DDSTORE_GROUP_MAP_FILE is not None else dequed
    master_addr = f"HYDRAGNN_MASTER_ADDR={trial_nodes[0]}"
    nodelist = ",".join(trial_nodes)
    handshake_dir = DDSTORE_HANDSHAKE_DIR
    if DDSTORE_GROUP_MAP_FILE is not None:
        with open(DDSTORE_GROUP_MAP_FILE) as group_map:
            groups = {
                frozenset(nodes.split(",")): path
                for nodes, path in (line.rstrip("\n").split("\t", 1) for line in group_map)
            }
        handshake_dir = groups.get(frozenset(trial_nodes))
        if handshake_dir is None:
            raise ValueError(f"No DDStore core group for trial nodes {nodelist}")
        print(f"Trial {trial.id}: nodes {nodelist} -> DDStore {handshake_dir}", flush=True, file=f)

    prefix = " ".join(
        [
            f"srun",
            f"-N {NNODES_PER_TRIAL} -n {NGPUS_PER_TRIAL}",
            f"--ntasks-per-node={GPUS_PER_NODE}"
            + ("" if HYDRAGNN_SYSTEM == "perlmutter" else f" --gpus-per-node={GPUS_PER_NODE}"),
            f"--cpus-per-task {OMP_NUM_THREADS} --threads-per-core 1 --cpu-bind {'cores' if HYDRAGNN_SYSTEM == 'perlmutter' else 'threads'}",
            f"--gpus-per-task=1 --gpu-bind={'none' if HYDRAGNN_SYSTEM == 'perlmutter' else 'closest'}",
            f"--export=ALL,{master_addr}",
            f"--nodelist={nodelist}",
            f"--output {DEEPHYPER_LOG_DIR}/output_{SLURM_JOB_ID}_{trial.id}.txt",
            f"--error {DEEPHYPER_LOG_DIR}/error_{SLURM_JOB_ID}_{trial.id}.txt",
            "--kill-on-bad-exit=1",
            *(
                ["--exact"]
                if HYDRAGNN_SYSTEM == "perlmutter"
                and DDSTORE_HANDSHAKE_DIR is not None
                and os.getenv("DDSTORE_COLOCATE") == "1"
                else []
            ),
        ]
    )
    if DDSTORE_HANDSHAKE_DIR is not None or HYDRAGNN_SYSTEM == "perlmutter":
        prefix = f"{prefix} {VNI_WRAP}"

    ## Launch time and trial id for the trial's timing JSON (HYDRAGNN_TIMING_DIR)
    prefix = (
        f"export TRIAL_LAUNCH_TIME=$(date +%s.%N) HYDRAGNN_TRIAL_ID={trial.id}; {prefix}"
    )
    inputfile = (
        os.path.join(os.path.dirname(__file__), "gfm_mlip.json")
        if HYDRAGNN_SYSTEM == "perlmutter"
        else "gfm_mlip.json"
    )
    command = " ".join(
        [
            prefix,
            python_exe,
            "-u",
            python_script,
            f"--inputfile={inputfile}",
            f"--multi",
            f"--ddstore",
            f"--precision=fp64",
            f"--multi_model_list={MULTI_MODEL_LIST}",
            f"--num_epoch={NUM_EPOCH}",
            f"--batch_size={BATCH_SIZE}",
            f"--num_samples={BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH}",
            f"--oversampling",
            f"--oversampling_num_samples={BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH}",
            f"--log={log_name}",
        ]
        + (
            [f"--ddstore_handshake_dir={handshake_dir}"]
            if handshake_dir is not None
            else []
        )
        + [
            f"--{param}={trial.parameters[param]}"
            for param in trial.parameters
            if param != "id"
        ]
    )

    if OMNISTAT_DIR is not None:
        ## Wrap command with omnistat annotate start/stop
        annotate_start = (
            f"{OMNISTAT_DIR}/omnistat-annotate --mode start --text 'gfm_{trial.id}'"
        )
        annotate_stop = f"{OMNISTAT_DIR}/omnistat-annotate --mode stop"

        # additional job step tracking commands
        job_step_start = f"{OMNISTAT_DIR}/omnistat-rms-env"
        job_step_stop = f"{OMNISTAT_DIR}/omnistat-rms-env --nostep"

        command = "; ".join(
            [annotate_start, job_step_start, command, job_step_stop, annotate_stop]
        )

    print("Command = ", command, flush=True, file=f)

    output = -math.inf
    num_pattern = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?|[-+]?(?:inf|nan)"
    trial_start = time.time()
    try:
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        fout = open(f"{DEEPHYPER_LOG_DIR}/error_{SLURM_JOB_ID}_{trial.id}.txt", "r")
        while True:
            line = fout.readline()
            if "Tasks Val Loss" in line:
                nums = re.findall(num_pattern, line, flags=re.IGNORECASE)
                ## Task Val Loss: [a, b, c]. The output must be -(b+c)/2
                output = -0.5 * (to_float(nums[1]) + to_float(nums[2]))
                print(
                    f"Val losses: {-to_float(nums[1])}, {-to_float(nums[2])} Average: {output}",
                    flush=True,
                    file=f,
                )
            if not line:
                break
        fout.close()
    except Exception as excp:
        print(excp, flush=True, file=f)
        output = -math.inf

    print("Output:", output, flush=True, file=f)
    objective = output
    print(objective, flush=True, file=f)
    metadata = {"trial_start": trial_start, "trial_end": time.time()}
    f.close()

    return {"objective": objective, "metadata": metadata}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mpnn_type", type=str, default="EGNN,SchNet,DimeNet,MACE,PAINN,PNAEq"
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=200,
        help="Number of max evaluations for HPO search",
    )
    args = parser.parse_args()
    mpnn_type_list = args.mpnn_type.split(",")

    log_name = f"gfm-{SLURM_JOB_ID}"
    if len(mpnn_type_list) == 1:
        log_name = f"gfm_{mpnn_type_list[0]}-{SLURM_JOB_ID}"
    log_name = os.getenv("HYDRAGNN_DEEPHYPER_DIR", log_name)

    from deephyper.hpo import HpProblem, CBO
    from deephyper.evaluator import ProcessPoolEvaluator, queued
    from hydragnn.utils.hpo.deephyper import read_node_list

    ## We maintain dict of hyperparameters and then problem.add later
    hyperparameters = dict()
    hyperparameters["num_conv_layers"] = (2, 6)
    # keep <=6 conv layers to mitigate oversmoothing/oversquashing
    # ~5B params target (rough):  problem.add_hyperparameter((2, 6), "num_conv_layers")
    # ~10B params target (rough): problem.add_hyperparameter((2, 6), "num_conv_layers")
    # ~100B params target (rough): problem.add_hyperparameter((2, 6), "num_conv_layers")

    hyperparameters["hidden_dim"] = (100, 3000)
    # compensate with channel width (MPNN hidden_dim) instead of extra depth
    # ~5B params target (rough):  problem.add_hyperparameter((3000, 10000), "hidden_dim")
    # ~10B params target (rough): problem.add_hyperparameter((10000, 20000), "hidden_dim")
    # ~100B params target (rough): problem.add_hyperparameter((20000, 30000), "hidden_dim")

    hyperparameters["num_headlayers"] = (2, 4)
    # keep <=4 MLP head layers to mitigate oversmoothing/oversquashing
    # ~5B params target (rough):  problem.add_hyperparameter((2, 4), "num_headlayers")
    # ~10B params target (rough): problem.add_hyperparameter((2, 4), "num_headlayers")
    # ~100B params target (rough): problem.add_hyperparameter((2, 4), "num_headlayers")

    hyperparameters["dim_headlayers"] = (300, 2000)
    # ~5B params target (rough):  problem.add_hyperparameter((2000, 4000), "dim_headlayers")
    # ~10B params target (rough): problem.add_hyperparameter((4000, 600), "dim_headlayers")
    # ~100B params target (rough): problem.add_hyperparameter((6000, 9000), "dim_headlayers")

    hyperparameters["force_weight"] = [10.0, 50.0, 100.0]
    hyperparameters["learning_rate"] = (1e-5, 1e-3)
    hyperparameters["mpnn_type"] = mpnn_type_list

    ## Model specific hyperparameters
    if len(mpnn_type_list) == 1:
        if mpnn_type_list[0] == "EGNN":
            hyperparameters["force_weight"] = (10.0, 1000.0)
            hyperparameters["learning_rate"] = (1e-5, 3e-3)
        elif mpnn_type_list[0] == "SchNet":
            hyperparameters["force_weight"] = (10.0, 1000.0)
            hyperparameters["learning_rate"] = (1e-5, 3e-3)

            hyperparameters["num_filters"] = (6, 300)
            hyperparameters["num_gaussians"] = (3, 100)
        elif mpnn_type_list[0] == "DimeNet":
            hyperparameters["hidden_dim"] = (10, 100)
            hyperparameters["force_weight"] = (10.0, 1000.0)
            hyperparameters["learning_rate"] = (1e-5, 3e-3)

            hyperparameters["basis_emb_size"] = (8, 200)
            hyperparameters["envelope_exponent"] = (4, 6)
            hyperparameters["int_emb_size"] = (16, 500)
            hyperparameters["out_emb_size"] = (24, 4000)
            hyperparameters["num_after_skip"] = (0, 2)
            hyperparameters["num_before_skip"] = (0, 2)
            hyperparameters["num_radial"] = (3, 12)
            hyperparameters["num_spherical"] = (2, 8)
        elif mpnn_type_list[0] == "MACE":
            hyperparameters["hidden_dim"] = (100, 1000)
            hyperparameters["force_weight"] = (10.0, 1000.0)
            hyperparameters["learning_rate"] = (1e-5, 3e-3)

            hyperparameters["radial_type"] = ["bessel", "gaussian", "chebyshev"]
            hyperparameters["distance_transform"] = ["Agnesi", "Soft"]
            hyperparameters["num_radial"] = (3, 12)
            hyperparameters["max_ell"] = (2, 3)
            hyperparameters["node_max_ell"] = (1, 3)
            hyperparameters["correlation"] = (2, 3)
        elif mpnn_type_list[0] == "PAINN":
            hyperparameters["hidden_dim"] = (100, 1000)
            hyperparameters["force_weight"] = (10.0, 1000.0)
            hyperparameters["learning_rate"] = (1e-5, 3e-3)

            hyperparameters["num_radial"] = (3, 12)
        elif mpnn_type_list[0] == "EGNN":
            hyperparameters["force_weight"] = (10.0, 1000.0)
            hyperparameters["learning_rate"] = (1e-5, 3e-3)
        elif mpnn_type_list[0] == "PNAEq":
            hyperparameters["hidden_dim"] = (100, 1000)
            hyperparameters["force_weight"] = (10.0, 1000.0)
            hyperparameters["learning_rate"] = (1e-5, 3e-3)

            hyperparameters["num_radial"] = (3, 12)
        else:
            raise ValueError(f"Unsupported MPNN type: {mpnn_type_list[0]}")

    ## Create HPO problem with the defined hyperparameters
    problem = HpProblem()
    for k, v in hyperparameters.items():
        problem.add_hyperparameter(v, k)

    # Create the node queue
    queue, _ = read_node_list()
    queue_pop_per_task = NNODES_PER_TRIAL
    if DDSTORE_GROUP_MAP_FILE is not None:
        with open(DDSTORE_GROUP_MAP_FILE) as group_map:
            queue = [
                tuple(nodes.split(","))
                for nodes, _ in (
                    line.rstrip("\n").split("\t", 1) for line in group_map
                )
            ]
        if (
            len(queue) != NUM_CONCURRENT_TRIALS
            or any(len(group) != NNODES_PER_TRIAL for group in queue)
            or sorted(node for group in queue for node in group)
            != sorted(read_node_list()[0])
        ):
            raise ValueError("DDStore groups must partition the allocated trial nodes")
        queue_pop_per_task = 1
    elif os.getenv("DDSTORE_COLOCATE") != "1":
        queue = [n for n in queue if n not in DDSTORE_CORE_NODES]
    print("The queue:", queue, len(queue))
    print("NNODES_PER_TRIAL", NNODES_PER_TRIAL)
    print("NUM_CONCURRENT_TRIALS", NUM_CONCURRENT_TRIALS)
    print("NGPUS_PER_TRIAL", NGPUS_PER_TRIAL)
    print("NTOTGPUS", NTOTGPUS)
    print(NTOTGPUS, NGPUS_PER_TRIAL, NTOTGPUS // NGPUS_PER_TRIAL, len(queue))

    evaluator = queued(ProcessPoolEvaluator)(
        run,
        num_workers=NUM_CONCURRENT_TRIALS,
        queue=queue,
        queue_pop_per_task=queue_pop_per_task,
    )

    search = CBO(
        problem,
        acq_func="UCB",
        multi_point_strategy="cl_min",
        random_state=42,
        log_dir=log_name,
    )

    ## Preload results from previous runs.
    ## gfm_dir is the directory where the preloaded results are stored.
    ## If there is only one MPNN type, we look for results in gfm_{mpnn_type} directory.
    ## Otherwise, we look for results in gfm directory.
    gfm_dir = "gfm"
    if len(mpnn_type_list) == 1:
        gfm_dir = f"gfm_{mpnn_type_list[0]}"

    df_list = list()
    files = glob.glob(os.path.join(gfm_dir, "*.csv"))
    for fname in files:
        try:
            df = pd.read_csv(fname, header=0)
            total_rows = len(df)
            df["objective"] = pd.to_numeric(df["objective"], errors="coerce")
            df = df.dropna(subset=["objective"])
            valid_rows = list()
            for i in range(len(df)):
                single = df.iloc[i : i + 1]
                try:
                    ## Remove rows that can cause the following error:
                    ## "Error in loading preloaded results: Not all points are within the bounds of the space."
                    search.fit_surrogate(df.iloc[i : i + 1])
                    valid_rows.append(i)
                except:
                    continue
            print(f"Checking {fname}: total {total_rows}, valid {len(valid_rows)}")
            df = df.iloc[valid_rows]
            df_list.append(df)
        except Exception as excp:
            print(f"Error loading {fname}:", excp)

    ## Create a clean search object again
    search = CBO(
        problem,
        acq_func="UCB",
        multi_point_strategy="cl_min",
        random_state=42,
        log_dir=log_name,
    )

    if len(df_list) > 0:
        try:
            preloaded_results = pd.concat(df_list, ignore_index=True)
            print(
                f"Loaded {len(preloaded_results)} preloaded results from {len(df_list)} files."
            )
            search.fit_surrogate(preloaded_results)
        except Exception as excp:
            print("Error in loading preloaded results:", excp)

    timeout = None
    stop_progress = threading.Event()
    progress_thread = None
    if HYDRAGNN_SYSTEM == "perlmutter":
        print(f"[DeepHyper] Starting {args.max_evals} evaluations; trial logs: {DEEPHYPER_LOG_DIR}", flush=True)
        progress_thread = threading.Thread(
            target=report_trial_progress, args=(stop_progress,), daemon=True
        )
        progress_thread.start()
    try:
        results = search.search(evaluator, max_evals=args.max_evals, timeout=timeout)
    finally:
        if progress_thread is not None:
            stop_progress.set()
            progress_thread.join()
    print(results)

    sys.exit(0)
