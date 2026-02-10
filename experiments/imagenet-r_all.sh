# bash experiments/imagenet-r.sh
# experiment settings
DATASET=ImageNet_R

# save directory
OUTDIR_S=outputs/${DATASET}/5-task
OUTDIR=outputs/${DATASET}/10-task
OUTDIR_L=outputs/${DATASET}/20-task

# training settings
GPUID='0'
REPEAT=5
OVERWRITE=1

###############################################################

# process inputs
mkdir -p $OUTDIR_S
mkdir -p $OUTDIR
mkdir -p $OUTDIR_L



# SMoPE
# prompt parameter args:
#    arg 1 = prompt length, equal to 2 * N_p, where N_p is the number of prompt experts
#    arg 2 = K, number of experts to use for each input
#    arg 3 = alpha_router
#    arg 4 = alpha_proto
#    arg 5 = epsilon

# --- 5-task ------
python -u run.py --config configs/imnet-r_prompt_short_smope.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name OnePrompt \
    --prompt_param 50 5 1e-4 1e-5 0.7 --seeds 0 1 2 3 4 \
    --crct_epochs 60 --ca_batch_size_ratio 2 \
    --log_dir ${OUTDIR_S}/one-prompt
sleep 10

# --- 10-task ------ 
python -u run.py --config configs/imnet-r_prompt_smope.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name VAPTPrompt \
    --prompt_param 50 5 5e-5 1e-4 0.5 --seeds 0 1 2 3 4 \
    --crct_epochs 50 --ca_batch_size_ratio 6 \
    --log_dir ${OUTDIR}/one-prompt
sleep 10

# --- 20-task ------
python -u run.py --config configs/imnet-r_prompt_long_smope.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name VAPTPrompt \
    --prompt_param 50 5 1e-4 5e-5 0.3 --seeds 0 1 2 3 4 \
    --crct_epochs 50 --ca_batch_size_ratio 4 \
    --log_dir ${OUTDIR_L}/one-prompt