#!/bin/bash

module load pytorch/2.13.0

export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=16
export CC=cc
export CXX=CC

source .venv/bin/activate
