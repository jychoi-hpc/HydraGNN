# DDStore method 2 core for the SC26 job scripts (source this file).
#
# A core srun step loads the data once and holds it in host memory; trial
# srun steps read it through gfm_mlip_all_mpnn.py --ddstore_handshake_dir.
# By default core and trials use separate nodes; DDSTORE_COLOCATE=1 puts
# both steps on the same nodes with disjoint CPU allocations (--exact).
# Core and trials are separate steps, so the job needs
# --network=single_node_vni,job_vni and every step runs through vni-wrap.sh.
#
# Needs: HYDRAGNN_ROOT, NNODES_PER_TRIAL, DEEPHYPER_LOG_DIR.
# Optional: GPUS_PER_NODE (default 8), OMP_NUM_THREADS (default 7),
# DDSTORE_CPU_BIND (default threads), DDSTORE_COLOCATE (default 0),
# DDSTORE_CORE_CPUS_PER_TASK (default OMP_NUM_THREADS),
# DDSTORE_CORE_NNODES (default NNODES_PER_TRIAL),
# DDSTORE_CORE_START_TIMEOUT_S (default 300).
#
#   ddstore_split_nodes     CORE_NODES = first NNODES_PER_TRIAL nodes and
#                           TRIAL_NODES = the rest, or both = all nodes with
#                           DDSTORE_COLOCATE=1; NNODES = trial node count
#   start_core DATA_ARGS    start the core and wait until it is ready; stops
#                           it on exit (trap)
#   stop_core               write the stop file and wait for the core

export HYDRAGNN_DDSTORE_HANDSHAKE_DIR=$HYDRAGNN_ROOT/ddstore_runtime/$SLURM_JOB_ID
VNI_WRAP=$HYDRAGNN_ROOT/examples/multidataset_hpo_sc26/vni-wrap.sh
CORE_PID=""

function ddstore_split_nodes() {
    local all_nodes
    mapfile -t all_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    if [ "${DDSTORE_COLOCATE:-0}" = "1" ]; then
        # The core step and each trial step share every allocated node.
        export CORE_NODES=$(IFS=,; echo "${all_nodes[*]}")
        export TRIAL_NODES="$CORE_NODES"
        export NNODES=${#all_nodes[@]}
    else
        export CORE_NODES=$(IFS=,; echo "${all_nodes[*]:0:$NNODES_PER_TRIAL}")
        export TRIAL_NODES=$(IFS=,; echo "${all_nodes[*]:$NNODES_PER_TRIAL}")
        export NNODES=$(( ${#all_nodes[@]} - NNODES_PER_TRIAL ))
    fi
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

# Core: by default the same rank count as a trial; DDSTORE_CORE_NNODES
# can override that count for a single core spanning multiple trials.
# --num_samples is per core rank. CPU only.
# --kill-on-bad-exit: one failed rank ends the step, so the wait for
# "ready" below fails fast instead of hanging.
function start_core() {
    local core_log=$DEEPHYPER_LOG_DIR/ddstore_core_$SLURM_JOB_ID.txt
    local colocate_args=()
    if [ "${DDSTORE_COLOCATE:-0}" = "1" ]; then
        colocate_args=(--exact --gres=none)
    fi
    rm -rf $HYDRAGNN_DDSTORE_HANDSHAKE_DIR
    mkdir -p $HYDRAGNN_DDSTORE_HANDSHAKE_DIR
    echo "DDStore handshake dir: $HYDRAGNN_DDSTORE_HANDSHAKE_DIR"

    srun -N ${DDSTORE_CORE_NNODES:-$NNODES_PER_TRIAL} -n $(( ${DDSTORE_CORE_NNODES:-$NNODES_PER_TRIAL} * ${GPUS_PER_NODE:-8} )) -w $CORE_NODES \
        --ntasks-per-node=${GPUS_PER_NODE:-8} --cpus-per-task ${DDSTORE_CORE_CPUS_PER_TASK:-${OMP_NUM_THREADS:-7}} --threads-per-core 1 --cpu-bind ${DDSTORE_CPU_BIND:-threads} \
        "${colocate_args[@]}" --export=ALL,OMP_NUM_THREADS=${DDSTORE_CORE_CPUS_PER_TASK:-${OMP_NUM_THREADS:-7}} \
        --kill-on-bad-exit=1 \
        -o $core_log -e $core_log \
        $VNI_WRAP python -u $HYDRAGNN_ROOT/examples/multidataset_hpo_sc26/gfm_mlip_all_mpnn.py \
        $1 --log=ddstore_core_$SLURM_JOB_ID \
        --ddstore_core --ddstore_handshake_dir=$HYDRAGNN_DDSTORE_HANDSHAKE_DIR &
    CORE_PID=$!
    # Stop the core on any exit, so a failed trial can't leave it running
    trap stop_core EXIT

    local waited=0
    local timeout=${DDSTORE_CORE_START_TIMEOUT_S:-300}
    while [ ! -f $HYDRAGNN_DDSTORE_HANDSHAKE_DIR/ready ]; do
        if ! kill -0 $CORE_PID 2>/dev/null; then
            echo "DDStore core exited before it was ready. Core log follows:"
            tail -50 $core_log
            CORE_PID=""
            return 1
        fi
        if [ "$waited" -ge "$timeout" ]; then
            echo "DDStore core not ready after ${waited}s. Core log follows:"
            tail -50 $core_log
            kill "$CORE_PID" 2>/dev/null || true
            wait "$CORE_PID" || true
            CORE_PID=""
            return 1
        fi
        sleep 5
        waited=$(( waited + 5 ))
    done
    echo "DDStore core ready after ${waited}s."
}

# Perlmutter DeepHyper: one independent MPI core group per NNODES_PER_TRIAL
# nodes, sharing those nodes with exactly one GPU trial at a time. The map
# associates a dequeued node group with its own handshake directory.
function stop_grouped_cores() {
    local i
    for ((i=0; i<${#GROUP_CORE_PIDS[@]}; i++)); do
        if kill -0 "${GROUP_CORE_PIDS[i]}" 2>/dev/null; then
            touch "${GROUP_CORE_DIRS[i]}/stop"
        fi
    done
    for ((i=0; i<${#GROUP_CORE_PIDS[@]}; i++)); do
        wait "${GROUP_CORE_PIDS[i]}" || true
    done
    GROUP_CORE_PIDS=()
    echo "DDStore core groups stopped."
}

function start_grouped_cores() {
    local data_args=$1
    local -a all_nodes group_nodes
    local i j nodes dir log waited ready_count=0
    local timeout=${DDSTORE_CORE_START_TIMEOUT_S:-300}
    GROUP_CORE_PIDS=()
    GROUP_CORE_DIRS=()
    mapfile -t all_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
    if (( ${#all_nodes[@]} == 0 || ${#all_nodes[@]} % NNODES_PER_TRIAL != 0 )); then
        printf 'ERROR: allocated nodes must divide into NNODES_PER_TRIAL groups.\n' >&2
        return 1
    fi
    export DDSTORE_GROUP_MAP_FILE="${HYDRAGNN_DDSTORE_HANDSHAKE_DIR}/groups.tsv"
    rm -rf "${HYDRAGNN_DDSTORE_HANDSHAKE_DIR}"
    mkdir -p "${HYDRAGNN_DDSTORE_HANDSHAKE_DIR}"
    : > "${DDSTORE_GROUP_MAP_FILE}"
    trap stop_grouped_cores EXIT

    for ((i=0; i<${#all_nodes[@]}; i+=NNODES_PER_TRIAL)); do
        group_nodes=("${all_nodes[@]:i:NNODES_PER_TRIAL}")
        nodes=$(IFS=,; printf '%s' "${group_nodes[*]}")
        j=$((i / NNODES_PER_TRIAL))
        dir="${HYDRAGNN_DDSTORE_HANDSHAKE_DIR}/group-${j}"
        log="${DEEPHYPER_LOG_DIR}/ddstore_core_${SLURM_JOB_ID}_group-${j}.txt"
        mkdir -p "${dir}"
        printf '%s\t%s\n' "${nodes}" "${dir}" >> "${DDSTORE_GROUP_MAP_FILE}"
        printf 'Starting DDStore core group %s on %s; handshake: %s\n' "${j}" "${nodes}" "${dir}"
        srun -N "${NNODES_PER_TRIAL}" -n "$((NNODES_PER_TRIAL * GPUS_PER_NODE))" -w "${nodes}" \
            --ntasks-per-node="${GPUS_PER_NODE}" --cpus-per-task="${DDSTORE_CORE_CPUS_PER_TASK:-4}" \
            --threads-per-core=1 --cpu-bind="${DDSTORE_CPU_BIND:-cores}" \
            --exact --gres=none \
            --export=ALL,OMP_NUM_THREADS="${DDSTORE_CORE_CPUS_PER_TASK:-4}",HYDRAGNN_TIMING_DIR="${HYDRAGNN_TIMING_DIR}/cores/group-${j}" \
            --kill-on-bad-exit=1 -o "${log}" -e "${log}" \
            "${VNI_WRAP}" python -u "${HYDRAGNN_ROOT}/examples/multidataset_hpo_sc26/gfm_mlip_all_mpnn.py" \
            ${data_args} --log="ddstore_core_${SLURM_JOB_ID}_group-${j}" \
            --ddstore_core --ddstore_handshake_dir="${dir}" &
        GROUP_CORE_PIDS+=("$!")
        GROUP_CORE_DIRS+=("${dir}")
    done

    waited=0
    while true; do
        ready_count=0
        for ((j=0; j<${#GROUP_CORE_PIDS[@]}; j++)); do
            if [[ -f "${GROUP_CORE_DIRS[j]}/ready" ]]; then
                ready_count=$((ready_count + 1))
            elif ! kill -0 "${GROUP_CORE_PIDS[j]}" 2>/dev/null; then
                printf 'ERROR: DDStore core group %s exited before ready. Log: %s\n' "${j}" "${DEEPHYPER_LOG_DIR}/ddstore_core_${SLURM_JOB_ID}_group-${j}.txt" >&2
                return 1
            fi
        done
        if (( ready_count == ${#GROUP_CORE_PIDS[@]} )); then
            printf 'All %s DDStore core groups ready after %ss.\n' "${ready_count}" "${waited}"
            return 0
        fi
        if (( waited >= timeout )); then
            printf 'ERROR: DDStore core groups not ready after %ss (%s/%s). Logs: %s/ddstore_core_%s_group-*.txt\n' \
                "${waited}" "${ready_count}" "${#GROUP_CORE_PIDS[@]}" "${DEEPHYPER_LOG_DIR}" "${SLURM_JOB_ID}" >&2
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
}
