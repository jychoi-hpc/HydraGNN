#!/bin/bash
# Slingshot VNI wrapper for DDStore core/reader steps on Frontier.
#
# With #SBATCH --network=job_vni every srun step sees
# SLINGSHOT_VNIS=<step VNI>,<job VNI>. The libfabric cxi provider uses only the
# first entry, so keep the job VNI (last entry) to let separate steps (DDStore
# core and trials) reach each other. See DDStore docs/hpc.md.
export SLINGSHOT_VNIS="${SLINGSHOT_VNIS##*,}"
exec "$@"
