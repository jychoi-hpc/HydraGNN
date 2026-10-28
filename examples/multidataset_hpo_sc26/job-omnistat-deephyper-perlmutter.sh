#!/bin/bash
#SBATCH -A m4828
#SBATCH -J HydraGNN-DeepHyper
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 02:00:00
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 16
#SBATCH --network=single_node_vni,job_vni

# Compare method 1 (per-trial store) against method 2 (one core per node group):
# COMPARE_TIMINGS=1 NNODES_PER_TRIAL=2 MAX_EVALS=4 ./job-omnistat-deephyper-perlmutter.sh
# Results: ./runs/m{1,2}_timing_<jobid>/ and ./runs/timing_compare_<jobid>.txt
# Set COMPARE_ORDER=ddstore_first to reverse the default plain_first order.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYDRAGNN_ROOT="${HYDRAGNN_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
MODULE_LOAD_SCRIPT="${MODULE_LOAD_SCRIPT:-${HYDRAGNN_ROOT}/module-to-load-perlmutter.sh}"
if [[ ! -f "${MODULE_LOAD_SCRIPT}" ]]; then
    printf 'ERROR: missing %s; run build-env-perlmutter.sh first.\n' "${MODULE_LOAD_SCRIPT}" >&2
    exit 1
fi

pushd "${HYDRAGNN_ROOT}" >/dev/null
# shellcheck disable=SC1090
source "${MODULE_LOAD_SCRIPT}"
popd >/dev/null
export PYTHONPATH="${HYDRAGNN_ROOT}:${PYTHONPATH:-}"
# The DeepHyper controller imports DeepSpeed too; keep its autotune cache
# node-local, just like the per-rank caches set by vni-wrap.sh.
export TRITON_CACHE_DIR="/tmp/triton-${SLURM_JOB_ID}-controller"
mkdir -p "${TRITON_CACHE_DIR}"
if ! python -c 'from deephyper.hpo import HpProblem, CBO; from deephyper.evaluator import ProcessPoolEvaluator, queued'; then
    printf 'ERROR: DeepHyper is missing; install it in .venv before submitting this job.\n' >&2
    exit 1
fi
python -c 'import adios2, pyddstore, torch; print("ADIOS2", adios2.__version__, "PyTorch", torch.__version__)'

export HYDRAGNN_SYSTEM=perlmutter
export GPUS_PER_NODE=4
export NNODES="${SLURM_JOB_NUM_NODES}"
export NNODES_PER_TRIAL="${NNODES_PER_TRIAL:-16}"
if (( NNODES_PER_TRIAL < 1 || NNODES % NNODES_PER_TRIAL != 0 )); then
    printf 'ERROR: NNODES_PER_TRIAL must divide %s allocated nodes.\n' "${NNODES}" >&2
    exit 1
fi
export NUM_CONCURRENT_TRIALS=$((NNODES / NNODES_PER_TRIAL))
export NTOTGPUS=$((NNODES * GPUS_PER_NODE))
export NGPUS_PER_TRIAL=$((NNODES_PER_TRIAL * GPUS_PER_NODE))
export NTOT_DEEPHYPER_RANKS="${NUM_CONCURRENT_TRIALS}"
# The interactive shell may inherit OMP_NUM_THREADS=1 from the PyTorch module.
export OMP_NUM_THREADS="${TRIAL_CPUS_PER_TASK:-8}"
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export PYTHONNOUSERSITE=1
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1
export HYDRAGNN_USE_FSDP=0
export HYDRAGNN_TRACE_LEVEL=1
export HYDRAGNN_MAX_NUM_BATCH="${HYDRAGNN_MAX_NUM_BATCH:-100}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export NUM_EPOCH="${NUM_EPOCH:-4}"
export HYDRAGNN_DDSTORE_METHOD="${HYDRAGNN_DDSTORE_METHOD:-2}"
if [[ "${HYDRAGNN_DDSTORE_METHOD}" != 1 && "${HYDRAGNN_DDSTORE_METHOD}" != 2 ]]; then
    printf 'ERROR: HYDRAGNN_DDSTORE_METHOD must be 1 or 2.\n' >&2
    exit 1
fi
unset DDSTORE_GROUP_MAP_FILE
export DDSTORE_FABRIC="${DDSTORE_FABRIC:-cxi}"
export HYDRAGNN_CUSTOM_DATALOADER=1
export HYDRAGNN_NUM_WORKERS=2
export MULTI_MODEL_LIST="${MULTI_MODEL_LIST:-Alexandria}"
IFS=, read -r -a models <<< "${MULTI_MODEL_LIST}"
if (( ${#models[@]} > NGPUS_PER_TRIAL )); then
    printf 'ERROR: %s datasets need at least that many trial ranks; this trial has %s. Set MULTI_MODEL_LIST to a shorter list.\n' "${#models[@]}" "${NGPUS_PER_TRIAL}" >&2
    exit 1
fi
for model in "${models[@]}"; do
    if [[ ! -e "${SCRIPT_DIR}/dataset/${model}-v2.bp" ]]; then
        printf 'ERROR: missing dataset %s. Set MULTI_MODEL_LIST to available datasets (e.g. Alexandria).\n' "${SCRIPT_DIR}/dataset/${model}-v2.bp" >&2
        exit 1
    fi
done
export DEEPHYPER_LOG_DIR="${DEEPHYPER_LOG_DIR:-${PWD}/deephyper-experiment-${SLURM_JOB_ID}}"
unset OMNISTAT_DIR

COMPARE_TIMINGS="${COMPARE_TIMINGS:-0}"
if [[ "${COMPARE_TIMINGS}" == 1 || "${HYDRAGNN_DDSTORE_METHOD}" == 2 ]]; then
    # Direct invocation inside salloc does not apply the #SBATCH directives.
    job_info="$(scontrol show job -o "${SLURM_JOB_ID}")"
    job_network="${job_info#* Network=}"
    job_network="${job_network%% *}"
    if [[ "${job_network}" != *job_vni* ]]; then
        printf 'ERROR: DDStore method 2 needs --network=single_node_vni,job_vni (allocation Network=%s).\n' "${job_network}" >&2
        exit 1
    fi
fi

export DDSTORE_CORE_CPUS_PER_TASK="${DDSTORE_CORE_CPUS_PER_TASK:-4}"
export DDSTORE_CPU_BIND=cores
if [[ "${COMPARE_TIMINGS}" == 1 || "${HYDRAGNN_DDSTORE_METHOD}" == 2 ]]; then
    if (( GPUS_PER_NODE * (DDSTORE_CORE_CPUS_PER_TASK + OMP_NUM_THREADS) > 64 )); then
        printf 'ERROR: core and trial request more than 64 physical CPU cores per Perlmutter node.\n' >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${SCRIPT_DIR}/ddstore-core.sh"
fi
CORE_DATA_ARGS="--inputfile=${SCRIPT_DIR}/gfm_mlip.json --multi --ddstore --precision=fp64 \
    --multi_model_list=${MULTI_MODEL_LIST} --batch_size=${BATCH_SIZE} \
    --num_samples=$((BATCH_SIZE * HYDRAGNN_MAX_NUM_BATCH)) --oversampling \
    --oversampling_num_samples=$((BATCH_SIZE * HYDRAGNN_MAX_NUM_BATCH))"
MAX_EVALS="${MAX_EVALS:-200}"

run_search() {
    export HYDRAGNN_DDSTORE_METHOD="$1"
    unset DDSTORE_GROUP_MAP_FILE HYDRAGNN_DDSTORE_HANDSHAKE_DIR DDSTORE_COLOCATE CORE_NODES
    mkdir -p "${DEEPHYPER_LOG_DIR}" "${HYDRAGNN_TIMING_DIR}"
    local t_start t_search_start t_search_end core_nodes=0
    t_start="$(date +%s.%N)"
    if [[ "${HYDRAGNN_DDSTORE_METHOD}" == 2 ]]; then
        export DDSTORE_COLOCATE=1
        export HYDRAGNN_DDSTORE_HANDSHAKE_DIR="${HYDRAGNN_ROOT}/ddstore_runtime/${SLURM_JOB_ID}"
        printf 'DDStore method 2: %s core groups of %s nodes (%s CPU ranks each); one %s GPU-rank trial per group.\n' \
            "${NUM_CONCURRENT_TRIALS}" "${NNODES_PER_TRIAL}" "${NGPUS_PER_TRIAL}" "${NGPUS_PER_TRIAL}"
        start_grouped_cores "${CORE_DATA_ARGS}"
        core_nodes="${NNODES}"
    fi
    t_search_start="$(date +%s.%N)"
    python -u "${SCRIPT_DIR}/gfm_deephyper_multi_all_mpnn.py" \
        --mpnn_type="${MPNN_TYPE:-SchNet}" --max_evals="${MAX_EVALS}"
    t_search_end="$(date +%s.%N)"
    if [[ "${HYDRAGNN_DDSTORE_METHOD}" == 2 ]]; then
        local i
        for ((i=0; i<NUM_CONCURRENT_TRIALS; i++)); do
            cp "${HYDRAGNN_DDSTORE_HANDSHAKE_DIR}/group-${i}/core_timing.json" \
                "${HYDRAGNN_TIMING_DIR}/core_timing_group-${i}.json"
        done
        stop_grouped_cores
        trap - EXIT
    fi
    cat > "${HYDRAGNN_TIMING_DIR}/search.json" <<EOF
{"method": ${HYDRAGNN_DDSTORE_METHOD}, "job_id": "${SLURM_JOB_ID}",
 "trial_nodes": ${NNODES}, "core_nodes": ${core_nodes}, "allocated_nodes": ${SLURM_JOB_NUM_NODES},
 "nnodes_per_trial": ${NNODES_PER_TRIAL}, "num_concurrent_trials": ${NUM_CONCURRENT_TRIALS},
 "max_evals": ${MAX_EVALS}, "t_start": ${t_start}, "t_search_start": ${t_search_start},
 "t_search_end": ${t_search_end}, "t_end": $(date +%s.%N)}
EOF
}

if [[ "${COMPARE_TIMINGS}" == 1 ]]; then
    COMPARE_ROOT="${COMPARE_ROOT:-${PWD}/runs}"
    case "${COMPARE_ORDER:-plain_first}" in
        plain_first) order=(1 2) ;;
        ddstore_first) order=(2 1) ;;
        *) printf 'ERROR: COMPARE_ORDER must be plain_first or ddstore_first.\n' >&2; exit 1 ;;
    esac
    for method in "${order[@]}"; do
        tag="m${method}_timing_${SLURM_JOB_ID}"
        printf '\nDeepHyper search: DDStore method %s (%s)\n' "${method}" "${tag}"
        export HYDRAGNN_RUN_TAG="${tag}"
        export DEEPHYPER_LOG_DIR="${COMPARE_ROOT}/${tag}/logs"
        export HYDRAGNN_DEEPHYPER_DIR="${COMPARE_ROOT}/${tag}/deephyper"
        export HYDRAGNN_TIMING_DIR="${COMPARE_ROOT}/${tag}/timings"
        run_search "${method}"
    done
    python "${SCRIPT_DIR}/compare-timings.py" \
        "${COMPARE_ROOT}/m1_timing_${SLURM_JOB_ID}" "${COMPARE_ROOT}/m2_timing_${SLURM_JOB_ID}" \
        --out "${COMPARE_ROOT}/timing_compare_${SLURM_JOB_ID}" | tee "${COMPARE_ROOT}/timing_compare_${SLURM_JOB_ID}.txt"
else
    export HYDRAGNN_TIMING_DIR="${HYDRAGNN_TIMING_DIR:-${DEEPHYPER_LOG_DIR}/timings}"
    run_search "${HYDRAGNN_DDSTORE_METHOD}"
fi
