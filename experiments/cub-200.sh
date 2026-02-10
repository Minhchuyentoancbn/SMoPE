# bash experiments/cub-200.sh
# experiment settings
DATASET=cub-200

# save directory
OUTDIR=outputs/${DATASET}/10-task

# hard coded inputs
GPUID='0'
REPEAT=5
OVERWRITE=1

###############################################################

# process inputs
mkdir -p $OUTDIR


# SMoPE
# prompt parameter args:
#    arg 1 = prompt length, equal to 2 * N_p, where N_p is the number of prompt experts
#    arg 2 = K, number of experts to use for each input
#    arg 3 = alpha_router
#    arg 4 = alpha_proto
#    arg 5 = epsilon

python -u run.py --config configs/cub-200_prompt_smope.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name OnePrompt \
    --prompt_param 50 5 1e-5 5e-5 0.4 --seeds 0 1 2 3 4 \
    --crct_epochs 50 --ca_batch_size_ratio 1 \
    --pretrained_weigh sup21k \
    --log_dir ${OUTDIR}/one-prompt