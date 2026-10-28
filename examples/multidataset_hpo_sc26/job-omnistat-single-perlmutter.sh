#!/bin/bash
#SBATCH -A m4828
#SBATCH -J HydraGNN-DDStore
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 02:00:00
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 16
#SBATCH --network=single_node_vni,job_vni

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYDRAGNN_ROOT="${HYDRAGNN_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
MODULE_LOAD_SCRIPT="${MODULE_LOAD_SCRIPT:-${HYDRAGNN_ROOT}/module-to-load-perlmutter.sh}"
if [[ ! -f "${MODULE_LOAD_SCRIPT}" ]]; then
    printf 'ERROR: missing %s; run build-env-perlmutter.sh first.\n' "${MODULE_LOAD_SCRIPT}" >&2
    exit 1
fi

cd "${HYDRAGNN_ROOT}"
# shellcheck disable=SC1090
source "${MODULE_LOAD_SCRIPT}"
export PYTHONPATH="${HYDRAGNN_ROOT}:${PYTHONPATH:-}"
python -c 'import adios2, pyddstore, torch; from pyddstore.torch import DistDatasetReader; print("ADIOS2", adios2.__version__, "PyTorch", torch.__version__)'
module list

export HYDRAGNN_SYSTEM=perlmutter
export GPUS_PER_NODE=4
export OMP_NUM_THREADS=8
export DDSTORE_CPU_BIND=cores
export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MPICH_GPU_SUPPORT_ENABLED=1
export PYTHONNOUSERSITE=1
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1
export HYDRAGNN_USE_FSDP=0
export HYDRAGNN_TRACE_LEVEL=1
export HYDRAGNN_MAX_NUM_BATCH=100
export HYDRAGNN_CUSTOM_DATALOADER=1
export HYDRAGNN_NUM_WORKERS=2
export HYDRAGNN_DDSTORE_METHOD="${HYDRAGNN_DDSTORE_METHOD:-2}"
export DDSTORE_FABRIC="${DDSTORE_FABRIC:-cxi}"
# One core rank per GPU rank, with separate CPUs on each training node.
export DDSTORE_COLOCATE=1
export DDSTORE_CORE_CPUS_PER_TASK=4
export NNODES_PER_TRIAL="${SLURM_JOB_NUM_NODES}"
export DEEPHYPER_LOG_DIR="${DEEPHYPER_LOG_DIR:-${SCRIPT_DIR}/deephyper-experiment-${SLURM_JOB_ID}}"
mkdir -p "${DEEPHYPER_LOG_DIR}"

if [[ "${HYDRAGNN_DDSTORE_METHOD}" == 2 ]]; then
    # HydraGNN's distributed core uses four ranks per node to publish each
    # trial's data. --exact keeps its CPUs separate from training CPUs.
    # When run directly inside salloc, #SBATCH directives above do not apply.
    job_info="$(scontrol show job -o "${SLURM_JOB_ID}")"
    job_network="${job_info#* Network=}"
    job_network="${job_network%% *}"
    if [[ "${job_network}" != *job_vni* ]]; then
        printf 'ERROR: DDStore method 2 needs a job VNI (allocation Network=%s).\n' "${job_network}" >&2
        printf 'Submit with sbatch, or allocate with salloc -N 2 -C gpu -A m4505 --qos=interactive -t 00:30:00 --network=single_node_vni,job_vni\n' >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${SCRIPT_DIR}/ddstore-core.sh"
    ddstore_split_nodes
elif [[ "${HYDRAGNN_DDSTORE_METHOD}" == 1 ]]; then
    export NNODES="${SLURM_JOB_NUM_NODES}"
else
    printf 'ERROR: HYDRAGNN_DDSTORE_METHOD must be 1 or 2.\n' >&2
    exit 1
fi

cleanup() {
    local status=$?
    trap - EXIT
    if [[ "${HYDRAGNN_DDSTORE_METHOD}" == 2 ]]; then
        stop_core || true
    fi
    exit "${status}"
}
trap cleanup EXIT

cd "${SCRIPT_DIR}"
# Keep the same data selection and fixed trial hyperparameters as job-omnistat-single.sh.
DATA_ARGS="--inputfile=gfm_mlip.json --multi --ddstore --precision=fp64 \
    --multi_model_list=Alexandria --batch_size=32 \
    --num_samples=3200 --oversampling --oversampling_num_samples=3200"
TRIAL_ARGS="--num_epoch=2 \
    --learning_rate=0.002439064005188516 \
    --dim_headlayers=986 \
    --force_weight=839.0144719977515 \
    --hidden_dim=2948 \
    --mpnn_type=SchNet \
    --num_conv_layers=6 \
    --num_filters=17 \
    --num_gaussians=43 \
    --num_headlayers=2"
SRUN_ARGS="--ntasks-per-node=4 --cpus-per-task 8 --threads-per-core=1 --cpu-bind=cores"

if [[ "${HYDRAGNN_DDSTORE_METHOD}" == 2 ]]; then
    NTRIALS="${NTRIALS:-2}"
    if (( NTRIALS < 1 )); then
        printf 'ERROR: NTRIALS must be at least 1.\n' >&2
        exit 1
    fi
    start_core "${DATA_ARGS}"
    # start_core sets its own EXIT trap; retain the job's cleanup handler.
    trap cleanup EXIT
    for ((i=0; i<NTRIALS; i++)); do
        printf '===== Trial %s =====\n' "${i}"
        time srun -N "${NNODES}" -n "$((NNODES * GPUS_PER_NODE))" -w "${TRIAL_NODES}" \
            ${SRUN_ARGS} --exact --gpus-per-task=1 --gpu-bind=none --kill-on-bad-exit=1 \
            "${VNI_WRAP}" python -u "${SCRIPT_DIR}/gfm_mlip_all_mpnn.py" \
            ${DATA_ARGS} ${TRIAL_ARGS} --log="gfm_${SLURM_JOB_ID}_${i}" \
            --ddstore_handshake_dir="${HYDRAGNN_DDSTORE_HANDSHAKE_DIR}"
    done
    stop_core
else
    time srun -N "${NNODES}" -n "$((NNODES * GPUS_PER_NODE))" \
        ${SRUN_ARGS} --gpus-per-task=1 --gpu-bind=none --kill-on-bad-exit=1 \
        bash -c 'export TRITON_CACHE_DIR="/tmp/triton-${SLURM_JOB_ID}-${SLURM_PROCID}"; mkdir -p "$TRITON_CACHE_DIR"; exec "$@"' _ \
        python -u "${SCRIPT_DIR}/gfm_mlip_all_mpnn.py" \
        ${DATA_ARGS} ${TRIAL_ARGS} --log="gfm_${SLURM_JOB_ID}_0"
fi
