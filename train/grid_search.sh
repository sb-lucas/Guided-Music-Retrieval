#!/bin/bash
# Grid search: trains every combination of the grid below, then evaluates all the runs on the validation set.
#
#   nohup bash grid_search.sh > grid_logs/main.log 2>&1 &
#
# - runs from the folder of this script (the cache/ folder is relative to it)
# - JOBS trainings run at the same time (the limit is the CPU of the DataLoaders, not the GPU)
# - the seeds are the outer loop: if you stop it early, the first seed already covers the whole grid
# - it can be restarted: the finished jobs are in grid_logs/done.txt and are skipped
# - the log of each training is grid_logs/<job>.log; the weights are in RUNS_FOLDER (.env), as for any training

set -u
cd "$(dirname "$(readlink -f "$0")")"

PYTHON=${PYTHON:-/home/ldsbrandao/exp/miniconda3/envs/tcc/bin/python}
JOBS=${JOBS:-6}
EPOCHS=${EPOCHS:-100}        # upper limit, the early stopping (patience 5) ends most runs before it
DATASET=mtsq

MODELS=(combiner concatenation)
SEEDS=(42 43 44)
BATCH_SIZES=(128 256 512)
BALANCES=(unbalanced balanced)
TAUS=(0.002 0.005 0.01)
LRS=(1e-4 3e-4 1e-3)

LOG_DIR=grid_logs
DONE_FILE=$LOG_DIR/done.txt
mkdir -p "$LOG_DIR"
touch "$DONE_FILE"

run_job() {
    local model=$1 seed=$2 batch_size=$3 balance=$4 tau=$5 lr=$6
    local key="${model}_seed${seed}_bs${batch_size}_${balance}_tau${tau}_lr${lr}"

    if grep -qx "$key" "$DONE_FILE"; then
        return 0
    fi

    local balance_flag=""
    if [ "$balance" = "balanced" ]; then
        balance_flag="--balance"
    fi

    # the run folder has the start time in seconds in its name, so two trainings never start in the same second
    flock "$LOG_DIR/launch.lock" sleep 1.2

    echo "[start]  $key"
    if "$PYTHON" -u train.py --dataset "$DATASET" --model "$model" --seed "$seed" --batch_size "$batch_size" \
            --tau "$tau" --lr "$lr" --epochs "$EPOCHS" $balance_flag > "$LOG_DIR/$key.log" 2>&1; then
        echo "$key" >> "$DONE_FILE"
        echo "[done]   $key"
    else
        echo "[FAILED] $key (see $LOG_DIR/$key.log)"
    fi
}
export -f run_job
export PYTHON EPOCHS DATASET LOG_DIR DONE_FILE

# one job per line: model seed batch_size balance tau lr
JOBS_FILE=$LOG_DIR/jobs.txt
: > "$JOBS_FILE"
for seed in "${SEEDS[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
        for balance in "${BALANCES[@]}"; do
            for model in "${MODELS[@]}"; do
                for tau in "${TAUS[@]}"; do
                    for lr in "${LRS[@]}"; do
                        echo "$model $seed $batch_size $balance $tau $lr" >> "$JOBS_FILE"
                    done
                done
            done
        done
    done
done

echo "$(wc -l < "$JOBS_FILE") jobs, $JOBS at a time, started $(date)"
xargs -P "$JOBS" -L 1 bash -c 'run_job "$@"' _ < "$JOBS_FILE"

echo "trainings finished $(date): $(wc -l < "$DONE_FILE") done"
echo "failed jobs are the [FAILED] lines above. Run this script again to retry them"

# every run in RUNS_FOLDER on the validation set (results in RESULTS_FOLDER, .env)
echo "evaluating on the validation set"
"$PYTHON" evaluate.py --dataset "$DATASET" --split val --zero_shot audio_only text_only sum --relevance exact > "$LOG_DIR/evaluate_val.log" 2>&1
echo "evaluation finished $(date), log in $LOG_DIR/evaluate_val.log"
