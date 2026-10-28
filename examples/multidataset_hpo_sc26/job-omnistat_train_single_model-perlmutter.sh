#!/bin/bash
#SBATCH -A m4828
#SBATCH -J HydraGNN
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 01:00:00
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH -c 32

function cmd() {
    echo "$@"
    time "$@"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYDRAGNN_ROOT="${HYDRAGNN_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
EXAMPLE_DIR="${HYDRAGNN_ROOT}/examples/multidataset_hpo_sc26"
MODULE_LOAD_SCRIPT="${MODULE_LOAD_SCRIPT:-${HYDRAGNN_ROOT}/module-to-load-perlmutter.sh}"

if [ ! -f "$MODULE_LOAD_SCRIPT" ]; then
    echo "ERROR: module load helper not found: $MODULE_LOAD_SCRIPT"
    echo "Run $HYDRAGNN_ROOT/build-env-perlmutter.sh first, or set MODULE_LOAD_SCRIPT explicitly."
    exit 1
fi

cd "$HYDRAGNN_ROOT" || exit 1
source "$MODULE_LOAD_SCRIPT"
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
echo "$LD_LIBRARY_PATH" | tr ':' '\n'

echo "===== HydraGNN envs ====="
export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MPICH_GPU_SUPPORT_ENABLED=1
export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=8
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1

env | grep ^MPICH || true
env | grep ^HYDRA || true

export HYDRAGNN_USE_FSDP=0
export HYDRAGNN_FSDP_VERSION=2
export HYDRAGNN_FSDP_STRATEGY=SHARD_GRAD_OP
export HYDRAGNN_TRACE_LEVEL=1
export HYDRAGNN_MAX_NUM_BATCH=100
export BATCH_SIZE=2
export NUM_EPOCH=4

export HYDRAGNN_DDSTORE_METHOD=1
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

#MULTI_MODEL_LIST=$datadir0,$datadir1,$datadir2,$datadir3,$datadir4,$datadir5,$datadir6,$datadir7,$datadir8,$datadir9,$datadir10,$datadir11,$datadir12,$datadir13,$datadir14,$datadir15
MULTI_MODEL_LIST=$datadir0

MASTER_HOST=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_IP=$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1 {print $1}')
if [ -z "$MASTER_IP" ]; then
    MASTER_IP="$MASTER_HOST"
fi
export MASTER_ADDR="$MASTER_IP"
export MASTER_PORT=${MASTER_PORT:-29501}
export HYDRAGNN_MASTER_ADDR=$MASTER_ADDR
export HYDRAGNN_MASTER_PORT=$MASTER_PORT

cmd srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*4)) -c32 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=none -l --kill-on-bad-exit=1 \
    --export=ALL \
    bash -c 'export TRITON_CACHE_DIR="/tmp/triton-${SLURM_JOB_ID}-${SLURM_PROCID}"; mkdir -p "$TRITON_CACHE_DIR"; exec "$@"' _ \
    python -u "$EXAMPLE_DIR/gfm_mlip_all_mpnn.py" \
    --log=multidataset_hpo-$SLURM_JOB_ID-NN$SLURM_JOB_NUM_NODES-PM-FSDP$HYDRAGNN_USE_FSDP --everyone \
    --inputfile="$EXAMPLE_DIR/gfm_mlip.json" --num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH)) \
    --oversampling --oversampling_num_samples=$((BATCH_SIZE*HYDRAGNN_MAX_NUM_BATCH)) \
    --multi --ddstore --multi_model_list=$MULTI_MODEL_LIST --batch_size=$BATCH_SIZE --num_epoch=$NUM_EPOCH \
    --precision=fp64 \
    --mpnn_type=SchNet \
    --num_conv_layers=6 \
    --hidden_dim=3000 \
    --num_headlayers=4 \
    --dim_headlayers=2000 \
    --learning_rate=0.001
