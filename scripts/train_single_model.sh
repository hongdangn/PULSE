#!/bin/bash

# wandb configurations
export WANDB__REQUIRE_LEGACY_SERVICE=TRUE
wandb login --relogin "810a71e03133ccddd00133f1fe9d2cd0f8001b4e"
export WANDB_ENTITY="hondang"
export WANDB_NAME="ddangdangnh"
export WANDB_PROJECT="ECGDistill"

# distributed training configurations
export GPUS_PER_NODE=1
export CUDA_VISIBLE_DEVICES=2
export NODE_RANK=0
export WORLD_SIZE=$(($GPUS_PER_NODE))

model_path="apple/FastVLM-1.5B"
version=qwen_2

data_path=/mnt/disk1/backup_user/dang.nh4/ecg_instruct/ECGInstruct_small.json
image_folder=/mnt/disk1/backup_user/dang.nh4/ecg_instruct/ecg/
output_dir=./output/fastvlm1.5B

num_epochs=3
BATCH_PER_GPU=2
GLOBAL_BATCH_SIZE=4

TOTAL_BATCH_SIZE=$(($WORLD_SIZE * $BATCH_PER_GPU))
GRAD_ACC_STEP=$(($GLOBAL_BATCH_SIZE / $TOTAL_BATCH_SIZE))

torchrun --nproc_per_node $GPUS_PER_NODE /home/dang.nh4/PULSE/LLaVA/llava/train/train_mem.py \
    --model_name_or_path $model_path \
    --version $version \
    --data_path $data_path \
    --image_folder $image_folder \
    --vision_tower mobileclip_l_1024\
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs $num_epochs \
    --per_device_train_batch_size $BATCH_PER_GPU \
    --per_device_eval_batch_size $BATCH_PER_GPU \
    --gradient_accumulation_steps $GRAD_ACC_STEP \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.05 \
    --save_total_limit 20 \
    --tune_mm_mlp_adapter True \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --lora_enable True 
