# DDStore method 2 core for the SC26 job scripts (source this file).
#
# A core srun step loads the data once and holds it in host memory; trial
# srun steps on other nodes read it over RDMA (gfm_mlip_all_mpnn.py
# --ddstore_handshake_dir) and come and go while the core keeps running.
# Core and trials are separate steps, so the job needs
# --network=single_node_vni,job_vni and every step runs through vni-wrap.sh.
#
# Needs: HYDRAGNN_ROOT, NNODES_PER_TRIAL, DEEPHYPER_LOG_DIR.
#
#   ddstore_split_nodes     CORE_NODES = first NNODES_PER_TRIAL nodes,
#                           TRIAL_NODES = the rest, NNODES = trial node count
#   start_core DATA_ARGS    start the core and wait until it is ready; stops
#                           it on exit (trap)
#   stop_core               write the stop file and wait for the core

export HYDRAGNN_DDSTORE_HANDSHAKE_DIR=$HYDRAGNN_ROOT/ddstore_runtime/$SLURM_JOB_ID
VNI_WRAP=$HYDRAGNN_ROOT/examples/multidataset_hpo_sc26/vni-wrap.sh
CORE_PID=""

function ddstore_split_nodes() {
    local all_nodes
    mapfile -t all_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    export CORE_NODES=$(IFS=,; echo "${all_nodes[*]:0:$NNODES_PER_TRIAL}")
    export TRIAL_NODES=$(IFS=,; echo "${all_nodes[*]:$NNODES_PER_TRIAL}")
    export NNODES=$(( ${#all_nodes[@]} - NNODES_PER_TRIAL ))
    echo "DDStore core nodes: $CORE_NODES"
    echo "DDStore trial nodes: $TRIAL_NODES"
}

function stop_core() {
    [ -n "$CORE_PID" ] || return 0
    touch $HYDRAGNN_DDSTORE_HANDSHAKE_DIR/stop
    wait $CORE_PID
    CORE_PID=""
    echo "DDStore core stopped."
}

# Core: same rank count as a trial (--num_samples is per rank), so it
# publishes the samples a method-1 trial would load. CPU only.
# --kill-on-bad-exit: one failed rank ends the step, so the wait for
# "ready" below fails fast instead of hanging.
function start_core() {
    local core_log=$DEEPHYPER_LOG_DIR/ddstore_core_$SLURM_JOB_ID.txt
    rm -rf $HYDRAGNN_DDSTORE_HANDSHAKE_DIR
    mkdir -p $HYDRAGNN_DDSTORE_HANDSHAKE_DIR
    echo "DDStore handshake dir: $HYDRAGNN_DDSTORE_HANDSHAKE_DIR"

    srun -N $NNODES_PER_TRIAL -n $(( NNODES_PER_TRIAL * 8 )) -w $CORE_NODES \
        --ntasks-per-node=8 --cpus-per-task 7 --threads-per-core 1 --cpu-bind threads \
        --kill-on-bad-exit=1 \
        -o $core_log -e $core_log \
        $VNI_WRAP python -u $HYDRAGNN_ROOT/examples/multidataset_hpo_sc26/gfm_mlip_all_mpnn.py \
        $1 --log=ddstore_core_$SLURM_JOB_ID \
        --ddstore_core --ddstore_handshake_dir=$HYDRAGNN_DDSTORE_HANDSHAKE_DIR &
    CORE_PID=$!
    # Stop the core on any exit, so a failed trial can't leave it running
    trap stop_core EXIT

    local waited=0
    while [ ! -f $HYDRAGNN_DDSTORE_HANDSHAKE_DIR/ready ]; do
        if ! kill -0 $CORE_PID 2>/dev/null; then
            echo "DDStore core exited before it was ready. Core log follows:"
            tail -50 $core_log
            CORE_PID=""
            exit 1
        fi
        sleep 5
        waited=$(( waited + 5 ))
    done
    echo "DDStore core ready after ${waited}s."
}
