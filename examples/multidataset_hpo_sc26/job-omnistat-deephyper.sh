#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J HydraGNN-DeepHyper
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 256
#SBATCH -C nvme
# DDStore method 2: the core and the trials are separate srun steps, which
# reach each other only through the job VNI (see vni-wrap.sh).
#SBATCH --network=single_node_vni,job_vni

function cmd() {
    echo "$@"
    time $@
}

function setup_bb()
{
    # Move a copy of the env to the NVMe on each node
    if [ -d /mnt/bb/${USER} ]; then
        srun -N${SLURM_JOB_NUM_NODES} --ntasks-per-node 1 -l -u df -h /mnt/bb/${USER} | grep -v Filesystem
        for FILENAME in HydraGNN-Installation-Frontier.tar.gz; do
            echo "Checking ${FILENAME}"
            if [ ! -f /mnt/bb/${USER}/${FILENAME} ]; then
                echo "Copying ${FILENAME} to each local NVME"
                time sbcast -pfv ${HYDRAGNN_ROOT}/${FILENAME} /mnt/bb/${USER}/${FILENAME}
                if [ ! "$?" == "0" ]; then
                    # CHECK EXIT CODE. When SBCAST fails, it may leave partial files on the compute nodes, and if you continue to launch srun,
                    # your application may pick up partially complete shared library files, which would give you confusing errors.
                    echo "SBCAST failed!"
                    break
                fi
            fi
            if [ ! -d /mnt/bb/${USER}/HydraGNN-Installation-Frontier ]; then
                echo "Untar ${FILENAME}"
                time srun -N${SLURM_JOB_NUM_NODES} --ntasks-per-node 1 -l -u tar -xf /mnt/bb/${USER}/${FILENAME} -C  /mnt/bb/${USER}/
            fi
        done
        echo "NVME is ready to use"
        srun -N${SLURM_JOB_NUM_NODES} --ntasks-per-node 1 -l -u df -h /mnt/bb/${USER} | grep -v Filesystem
    fi
}

HYDRAGNN_ROOT=/lustre/orion/world-shared/lrn070/jyc/frontier/HydraGNN-ddstore2

# Load conda environment
module reset
ml cpe/24.07
ml cce/18.0.0
ml rocm/7.13.0
ml amd-mixed/7.13.0
ml rccl-net-plugin/1.0     # aws-ofi-nccl RCCL net plugin (required for NCCL_NET="AWS Libfabric")
ml craype-accel-amd-gfx90a
ml PrgEnv-gnu
ml miniforge3/23.11.0-0
module unload darshan-runtime

setup_bb
if [ -d /mnt/bb/${USER}/HydraGNN-Installation-Frontier ]; then
    source activate /mnt/bb/${USER}/HydraGNN-Installation-Frontier-ROCm713/hydragnn_venv_rocm713
else
    source activate $HYDRAGNN_ROOT/HydraGNN-Installation-Frontier-ROCm713/hydragnn_venv_rocm713
fi

# Add HydraGNN in PYTHONPATH
export PYTHONPATH=$HYDRAGNN_ROOT:$PYTHONPATH

echo ""
echo "===== Check ====="
which python
python -c "import adios2; print(adios2.__version__, adios2.__file__)"
python -c "import torch; print(torch.__version__, torch.__file__)"

echo ""
echo "===== Module List ====="
module list

echo ""
echo "===== Check LD_LIBRARY_PATH ====="
echo $LD_LIBRARY_PATH  | tr ':' '\n'

echo "===== Performance envs ====="
# export PLUGIN_PATH=/ccs/sw/crusher/amdsw/aws-ofi-nccl/aws-ofi-nccl
# export LD_LIBRARY_PATH=${PLUGIN_PATH}/lib:${LD_LIBRARY_PATH}

export FI_MR_CACHE_MONITOR=kdreg2     # Required to avoid a deadlock.
export FI_CXI_DEFAULT_CQ_SIZE=131072  # Ask the network stack to allocate additional space to process message completions.
export FI_CXI_DEFAULT_TX_SIZE=2048    # Ask the network stack to allocate additional space to hold pending outgoing messages.
export FI_CXI_RX_MATCH_MODE=hybrid    # Allow the network stack to transition to software mode if necessary.
export FI_CXI_RDV_PROTO=alt_read
export FI_CXI_DISABLE_HOST_REGISTER=1

# export NCCL_NET_PLUGIN=${PLUGIN_PATH}/lib/librccl-net.so
export NCCL_NET_GDR_LEVEL="PHB"       # Typically improves performance, but remove this setting if you encounter a hang/crash.
export NCCL_CROSS_NIC=1               # On large systems, this NCCL setting has been found to improve performance
export NCCL_SOCKET_IFNAME=hsn0        # NCCL/RCCL will use the high speed network to coordinate startup.
export NCCL_NET="AWS Libfabric"

export TORCH_NCCL_HIGH_PRIORITY=1     # Use high priority stream for the NCCL/RCCL Communicator.
export GPU_MAX_HW_QUEUES=2

export HSA_FORCE_FINE_GRAIN_PCIE=1

# below are optional to debug RCCL stuff
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT

# The following have been found to help avoid hangs, but are not yet
# documented elsewhere
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0

echo "===== HydraGNN envs ====="
export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp
export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=7
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1

## Checking
env | grep ROCM
env | grep ^MI
env | grep ^MPICH
env | grep ^HYDRA

export HYDRAGNN_USE_FSDP=0
# export HYDRAGNN_FSDP_VERSION=2
# export HYDRAGNN_FSDP_STRATEGY=SHARD_GRAD_OP
export HYDRAGNN_TRACE_LEVEL=1
export HYDRAGNN_MAX_NUM_BATCH=100
export BATCH_SIZE=16 #200
export NUM_EPOCH=${NUM_EPOCH:-20}

# 2: a core srun step holds the data for the whole search and trial steps
# read it; 1: every trial loads its own DDStore.
export HYDRAGNN_DDSTORE_METHOD=${HYDRAGNN_DDSTORE_METHOD:-2}
export DDSTORE_FABRIC=${DDSTORE_FABRIC:-cxi}   # native Slingshot provider (default: hsn, tcp;ofi_rxm)
# CPU->NIC map for DDStore FABRIC_IFACE selection; computed once on the compute node
export DDSTORE_NIC_MAP=$(python3 -m cpu_nic_map --env)
# Do not set FI_HMEM=system here: RCCL's OFI plugin needs libfabric's ROCr
# HMEM support for GPU buffers (job 5776561 failed in NET/OFI with it).
export HYDRAGNN_CUSTOM_DATALOADER=1
export HYDRAGNN_NUM_WORKERS=2

# Dataset ordering matches gfm_deephyper_multi_all_mpnn.py multi_model_list
export datadir0=Alexandria
export datadir1=ANI1x
export datadir2=MPTrj
export datadir3=OC2020
export datadir4=OC2022
export datadir5=OC25
export datadir6=ODAC23
export datadir7=OMat24
export datadir8=OMol25
export datadir9=OMol25-neutral
export datadir10=OMol25-non-neutral
export datadir11=OPoly2026
export datadir12=Nabla2DFT
export datadir13=QCML
export datadir14=QM7X
export datadir15=transition1x

# export MULTI_MODEL_LIST=$datadir0,$datadir1,$datadir2,$datadir3,$datadir4,$datadir5,$datadir6,$datadir7,$datadir8,$datadir9,$datadir10,$datadir11,$datadir12,$datadir13,$datadir14,$datadir15
export MULTI_MODEL_LIST=${MULTI_MODEL_LIST:-$datadir0}

# COMPARE_TIMINGS=1: run a method-1 search, then a method-2 search on the
# same trial nodes, and compare their timing JSONs (compare-timings.py).
# The core nodes stay idle during the method-1 search.
# COMPARE_ORDER=ddstore_first runs the method-2 search first.
COMPARE_TIMINGS=${COMPARE_TIMINGS:-0}
COMPARE_ORDER=${COMPARE_ORDER:-plain_first}

# Configuration
export NNODES_PER_TRIAL=${NNODES_PER_TRIAL:-2} #256
if [ "$HYDRAGNN_DDSTORE_METHOD" = "2" ] || [ "$COMPARE_TIMINGS" = "1" ]; then
    source $HYDRAGNN_ROOT/examples/multidataset_hpo_sc26/ddstore-core.sh
    ddstore_split_nodes
else
    export NNODES=$SLURM_JOB_NUM_NODES # e.g., 100 total nodes
fi
export NUM_CONCURRENT_TRIALS=$(( $NNODES / $NNODES_PER_TRIAL ))
export NTOTGPUS=$(( $NNODES * 8 )) # e.g., 800 total GPUs
export NGPUS_PER_TRIAL=$(( 8 * $NNODES_PER_TRIAL )) # e.g., 32 GPUs per training
export NTOT_DEEPHYPER_RANKS=$(( $NTOTGPUS / $NGPUS_PER_TRIAL )) # e.g., 25 total DH ranks
export OMP_NUM_THREADS=7 # e.g., 8 threads per rank
[ $NTOTGPUS -ne $(($NGPUS_PER_TRIAL*$NUM_CONCURRENT_TRIALS)) ] && echo "ERROR!!"

# DeepHyper variables
export DEEPHYPER_LOG_DIR="deephyper-experiment"-$SLURM_JOB_ID
mkdir -p $DEEPHYPER_LOG_DIR

# (A) Setup omnistat sampling environment
ml use /sw/frontier/amdsw/modulefiles/
ml omnistat-wrapper
export OMNISTAT_DIR=$OMNISTAT_DIR
export OMNISTAT_CONFIG=$HYDRAGNN_ROOT/omnistat.hydragnn-external-fp64.config

# (B) Enable data collectors and polling (1 sec interval)
${OMNISTAT_WRAPPER} usermode --start --interval 15

# [ -z $MPNN_TYPE ] && MPNN_TYPE=EGNN,SchNet,DimeNet,MACE,PAINN,PNAEq
[ -z $MPNN_TYPE ] && MPNN_TYPE=SchNet

# Same data options as the trials (gfm_deephyper_multi_all_mpnn.py run())
CORE_DATA_ARGS="--inputfile=gfm_mlip.json --multi --ddstore --precision=fp64 \
    --multi_model_list=$MULTI_MODEL_LIST --batch_size=$BATCH_SIZE \
    --num_samples=$(( BATCH_SIZE * HYDRAGNN_MAX_NUM_BATCH )) --oversampling \
    --oversampling_num_samples=$(( BATCH_SIZE * HYDRAGNN_MAX_NUM_BATCH ))"

# One DeepHyper search with DDStore method $1, writing trial logs to
# $DEEPHYPER_LOG_DIR, timing JSONs to $HYDRAGNN_TIMING_DIR and the start/end
# times and node counts to $HYDRAGNN_TIMING_DIR/search.json
function run_search() {
    export HYDRAGNN_DDSTORE_METHOD=$1
    mkdir -p $DEEPHYPER_LOG_DIR $HYDRAGNN_TIMING_DIR
    local t_start=$(date +%s.%N) core_nodes=0
    if [ "$HYDRAGNN_DDSTORE_METHOD" = "2" ]; then
        start_core "$CORE_DATA_ARGS"
        core_nodes=$NNODES_PER_TRIAL
    fi
    local t_search_start=$(date +%s.%N)
    cmd python -u $HYDRAGNN_ROOT/examples/multidataset_hpo_sc26/gfm_deephyper_multi_all_mpnn.py \
        --mpnn_type=$MPNN_TYPE --max_evals=${MAX_EVALS:-$((NUM_CONCURRENT_TRIALS*2))}
    local t_search_end=$(date +%s.%N)
    if [ "$HYDRAGNN_DDSTORE_METHOD" = "2" ]; then
        stop_core
    fi
    cat > $HYDRAGNN_TIMING_DIR/search.json <<EOF2
{"method": $HYDRAGNN_DDSTORE_METHOD, "tag": "${HYDRAGNN_RUN_TAG:-}", "job_id": "$SLURM_JOB_ID",
 "trial_nodes": $NNODES, "core_nodes": $core_nodes, "nnodes_per_trial": $NNODES_PER_TRIAL,
 "num_concurrent_trials": $NUM_CONCURRENT_TRIALS, "max_evals": ${MAX_EVALS:-$((NUM_CONCURRENT_TRIALS*2))},
 "t_start": $t_start, "t_search_start": $t_search_start, "t_search_end": $t_search_end, "t_end": $(date +%s.%N)}
EOF2
}

if [ "$COMPARE_TIMINGS" = "1" ]; then
    RUNS_DIR=$HYDRAGNN_ROOT/runs
    case "$COMPARE_ORDER" in
        plain_first) ORDER="1 2" ;;
        ddstore_first) ORDER="2 1" ;;
        *) echo "COMPARE_ORDER must be plain_first or ddstore_first"; exit 1 ;;
    esac
    step=1
    for m in $ORDER; do
        tag=m${m}_timing_$SLURM_JOB_ID
        echo ""
        echo "[$step/3] DeepHyper search, DDStore method $m ($tag)"
        export HYDRAGNN_RUN_TAG=$tag
        export DEEPHYPER_LOG_DIR=$RUNS_DIR/$tag/logs
        export HYDRAGNN_DEEPHYPER_DIR=$RUNS_DIR/$tag/deephyper
        export HYDRAGNN_TIMING_DIR=$RUNS_DIR/$tag/timings
        run_search $m
        step=$(( step + 1 ))
    done
    echo ""
    echo "[3/3] Compare timings"
    python $HYDRAGNN_ROOT/examples/multidataset_hpo_sc26/compare-timings.py \
        $RUNS_DIR/m1_timing_$SLURM_JOB_ID $RUNS_DIR/m2_timing_$SLURM_JOB_ID \
        --out $RUNS_DIR/timing_compare_$SLURM_JOB_ID | tee $RUNS_DIR/timing_compare_$SLURM_JOB_ID.txt
else
    export HYDRAGNN_TIMING_DIR=${HYDRAGNN_TIMING_DIR:-$DEEPHYPER_LOG_DIR/timings}
    run_search $HYDRAGNN_DDSTORE_METHOD
fi

# (C) End of job: stop data collection
${OMNISTAT_WRAPPER} usermode --stop
