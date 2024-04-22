#!/bin/bash
#SBATCH -A CPH161
#SBATCH -J HydraGNN
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 8
#SBATCH -S 1

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES
NN=$JOBSIZE
NP=$((JOBSIZE*8))

cleanup () {
    rm -f *.lock
    srun -n$JOBSIZE -c1 --ntasks-per-node=1 bash -c "pkill -f python"
    srun -n$JOBSIZE -c1 --ntasks-per-node=1 bash -c "rm -f /dev/mqueue/* /dev/shm/ddstore* /dev/shm/sem.ddstore*"
    srun -n$JOBSIZE -c1 --ntasks-per-node=1 bash -c "ls /dev/mqueue/* /dev/shm/*"
    sleep 3
}

export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
#export MPICH_GPU_SUPPORT_ENABLED=1
#export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
# export MPICH_OFI_NIC_POLICY=GPU
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple

export DDSTORE_VERBOSE=0
export DDSTORE_ATTR_MSG_SIZE=2048
export DDSTORE_ATTR_MAX_MSG=10
export DDSTORE_MAX_NDCHANNEL=4

export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_MAX_NUM_BATCH=100
export HYDRAGNN_VALTEST=0
export NEPOCH=2

export SCOREP_ENABLE_PROFILING=false
export SCOREP_ENABLE_TRACING=true
export SCOREP_TOTAL_MEMORY=512M
SCOREP_OPT="-m scorep --verbose --keep-files --noinstrumenter --mpp=mpi"

echo $LD_LIBRARY_PATH  | tr ':' '\n'

env | grep ROCM
env | grep ^MI
env | grep ^MPICH
env | grep ^HYDRA

[ -z $TAG ] && TAG=-2M
[ -z $WIDTH ] && WIDTH=0
[ -z $M ] && M=SMALL
[ -z $X ] && X=1
[ -z $NW ] && NW=0
export HYDRAGNN_NUM_WORKERS=$NW
[ -z $SP ] && SP=0
[ $SP -eq 0 ] && SCOREP_OPT=""
[ $SP -eq 1 ] && export HYDRAGNN_MAX_NUM_BATCH=50 && export HYDRAGNN_VALTEST=0 && export NEPOCH=2

PRODUCER="ddstore-service.py --modelname=OC2020$TAG --producer --epochs=$NEPOCH"
CONSUMER="train.py --modelname=OC2020$TAG --inputfile=${M}_MTL.json --adios --ddstore --epochs=$NEPOCH --everyone --log=exp-OC2020$TAG-$SLURM_JOB_ID-NN$NN-NW$HYDRAGNN_NUM_WORKERS-M$M-W$WIDTH-X$X"

NN=$SLURM_JOB_NUM_NODES
NP=$NP

set -x

if [ $X -eq 0 ]; then
## For single run:
cleanup
export SCOREP_EXPERIMENT_DIRECTORY=scorep-$JOBID-x$X-n$HYDRAGNN_NUM_WORKERS
srun -N$NN -n$NP -c7 --gpus-per-task=1 --gpu-bind=closest \
    python -u train.py --modelname=OC2020$TAG --inputfile=${M}_MTL.json --adios --epochs=$NEPOCH --everyone --log=exp-OC2020$TAG-$SLURM_JOB_ID-NN$NN-NW$HYDRAGNN_NUM_WORKERS-M$M-W$WIDTH-X$X
fi

if [ $X -eq 1 ]; then
## For single run:
cleanup
export SCOREP_EXPERIMENT_DIRECTORY=scorep-$JOBID-x$X-n$HYDRAGNN_NUM_WORKERS
(time srun -N$NN -n$NP -c7 --gpus-per-task=1 --gpu-bind=closest python -u $CONSUMER) 2>&1 | tee run-X$X.log
fi

if [ $X -eq 2 ]; then
## For single run:
cleanup
MASTER_PORT=8889 srun -N$NN -n$NP -c1 --gpus-per-task=0 python -u $PRODUCER --mq 2>&1 > run-X$X-role0.log &
sleep 10
export SCOREP_EXPERIMENT_DIRECTORY=scorep-$JOBID-x$X-n$HYDRAGNN_NUM_WORKERS
(time MASTER_PORT=8890 srun -N$NN -n$NP -c6 --gpus-per-task=1 --gpu-bind=closest python -u $CONSUMER --mq) 2>&1 | tee run-X$X-role1.log
fi

if [ $X -eq 3 ]; then
## For producer-consumer stream run:
cleanup
MASTER_PORT=8889 srun -N$NN -n$NP -c1 --gpus-per-task=0 python -u $PRODUCER --mq --stream --nchannels=$HYDRAGNN_NUM_WORKERS 2>&1 > run-X$X-role0.log &
sleep 10
export SCOREP_EXPERIMENT_DIRECTORY=scorep-$JOBID-x$X-n$HYDRAGNN_NUM_WORKERS
(time MASTER_PORT=8890 srun -N$NN -n$NP -c6 --gpus-per-task=1 --gpu-bind=closest python -u $SCOREP_OPT $CONSUMER --mq --stream) 2>&1 | tee run-X$X-role1.log
fi

if [ $X -eq 4 ]; then
## For producer-consumer shmemq run:
cleanup
MASTER_PORT=8889 srun -N$NN -n$NP -c1 --gpus-per-task=0 python -u $PRODUCER --mq --shmemq --nchannels=$HYDRAGNN_NUM_WORKERS 2>&1 > run-X$X-role0.log &
sleep 10
export SCOREP_EXPERIMENT_DIRECTORY=scorep-$JOBID-x$X-n$HYDRAGNN_NUM_WORKERS
(time MASTER_PORT=8890 srun -N$NN -n$NP -c6 --gpus-per-task=1 --gpu-bind=closest python -u $SCOREP_OPT $CONSUMER --mq --shmemq) 2>&1 | tee run-X$X-role1.log
fi

if [ $X -eq 12 ]; then
## For producer-consumer run:
[ $ROLE -eq 1 ] && MASTER_PORT=8889 srun -N$NN -n$NP -c1 --gpus-per-task=0 python -u $PRODUCER --mq 2>&1 | tee run-X$X-role0.log
[ $ROLE -eq 0 ] && (time MASTER_PORT=8890 srun -N$NN -n$NP -c6 --gpus-per-task=1 --gpu-bind=closest python -u $CONSUMER --mq) 2>&1 | tee run-X$X-role1.log
fi

if [ $X -eq 13 ]; then
## For producer-consumer stream run:
[ $ROLE -eq 1 ] && MASTER_PORT=8889 srun -N$NN -n$NP -c1 --gpus-per-task=0 -u python -u $PRODUCER --mq --stream --nchannels=$HYDRAGNN_NUM_WORKERS 2>&1 | tee run-X$X-role0.log
sleep 10
[ $ROLE -eq 0 ] && (time MASTER_PORT=8890 srun -N$NN -n$NP -c6 --gpus-per-task=1 --gpu-bind=closest -u python -u $CONSUMER --mq --stream) 2>&1 | tee run-X$X-role1.log
fi

if [ $X -eq 14 ]; then
## For producer-consumer stream run:
[ $ROLE -eq 1 ] && MASTER_PORT=8889 srun -N$NN -n$NP -c1 --gpus-per-task=0 -u python -u $PRODUCER --mq --shmemq --nchannels=$HYDRAGNN_NUM_WORKERS 2>&1 | tee run-X$X-role0.log
sleep 10
[ $ROLE -eq 0 ] && (time MASTER_PORT=8890 srun -N$NN -n$NP -c6 --gpus-per-task=1 --gpu-bind=closest -u python -u $CONSUMER --mq --shmemq) 2>&1 | tee run-X$X-role1.log
fi


set +x
