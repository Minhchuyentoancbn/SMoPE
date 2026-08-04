# bash experiments/cub-200.sh
set -euo pipefail

# experiment settings
DATASET=cub-200

# save directory
RUN_ID=$(date +%Y%m%d_%H%M%S)
OUTROOT=outputs/${DATASET}/10-task
OUTDIR=${OUTROOT}/${RUN_ID}
LOGDIR=${OUTDIR}/one-prompt

# hard coded inputs
GPUID='0'
REPEAT=2
OVERWRITE=1

###############################################################

# process inputs
mkdir -p "$OUTDIR"
mkdir -p "$LOGDIR"

# environment snapshot
{
  echo "run_id=${RUN_ID}"
  echo "start_time=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "hostname=$(hostname)"
  echo "workdir=$(pwd)"
  echo "git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "python=$(python -V 2>&1 || true)"
  echo "pip=$(python -m pip --version 2>&1 || true)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
} > "${OUTDIR}/run_info.txt"

python -m pip freeze > "${OUTDIR}/pip_freeze.txt" 2>/dev/null || true
nvidia-smi > "${OUTDIR}/nvidia_smi_start.txt" 2>/dev/null || true

# SMoPE
# prompt parameter args:
#    arg 1 = prompt length, equal to 2 * N_p, where N_p is the number of prompt experts
#    arg 2 = K, number of experts to use for each input
#    arg 3 = alpha_router
#    arg 4 = alpha_proto
#    arg 5 = epsilon

TRAIN_STATUS=0
python -u run.py --config configs/cub-200_prompt_smope.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name OnePrompt \
    --prompt_param 50 5 1e-5 5e-5 0.4 --seeds 0 1 \
    --crct_epochs 50 --ca_batch_size_ratio 1 \
    --pretrained_weight sup21k \
    --log_dir "${LOGDIR}" || TRAIN_STATUS=$?

nvidia-smi > "${OUTDIR}/nvidia_smi_end.txt" 2>/dev/null || true
{
  echo "end_time=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "train_status=${TRAIN_STATUS}"
} >> "${OUTDIR}/run_info.txt"

tar -czf "${OUTDIR}/one-prompt.tar.gz" -C "${OUTDIR}" one-prompt

exit ${TRAIN_STATUS}
