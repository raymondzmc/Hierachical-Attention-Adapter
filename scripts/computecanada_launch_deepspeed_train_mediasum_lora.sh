#!/bin/bash
#SBATCH --gres=gpu:a100:4       # Request GPU "generic resources"
#SBATCH --cpus-per-task=32  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=192000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=2-0:00
#SBATCH --output=mediasum-lora.out
module load StdEnv/2023 arrow/15 python/3.10
source ~/projects/def-carenini/liraymo6/envs/project/bin/activate

accelerate launch --config_file=./deepspeed_zero3.yaml \
train.py --dataset_name mediasum --use_lora --num_train_epochs 1 --batch_size 1 \
         --save_strategy steps --save_steps 0.25 --evaluation_strategy no \
         --gradient_accumulation_steps 16 --gradient_checkpointing --learning_rate 1e-4 \
         --output_dir $OUTPUT_DIR \
         --do_train >> "$OUTPUT_DIR/ds_train_log.txt"