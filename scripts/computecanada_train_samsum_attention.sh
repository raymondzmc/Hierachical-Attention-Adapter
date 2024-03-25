#!/bin/bash
#SBATCH --gres=gpu:a100:4       # Request GPU "generic resources"
#SBATCH --cpus-per-task=16  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G      # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=6:00:00
#SBATCH --output=samsum-attention.out
module load StdEnv/2023 arrow/15 python/3.10
source ~/projects/def-carenini/liraymo6/envs/project/bin/activate

export HF_HOME="/home/liraymo6/scratch/.cache"
export HF_DATASETS_CACHE="/home/liraymo6/scratch/.cache/datasets"
export HF_DATASETS_OFFLINE=1
export OUTPUT_DIR='output/mistral-samsum-attention'
mkdir -p $OUTPUT_DIR

accelerate launch --config_file=./deepspeed_zero3.yaml train.py --dataset_name samsum \
         --adapter_method attention --injection_location sa --num_layers 32 --use_last_layer --gate_type tanh \
         --num_train_epochs 3 --batch_size 1 --gradient_accumulation_steps 16 \
         --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
         --output_dir $OUTPUT_DIR --do_train >> "$OUTPUT_DIR/train_log.txt"

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name samsum \
         --adapter_method attention --injection_location sa --num_layers 32 --use_last_layer --gate_type tanh \
         --eval_batch_size 1 --output_dir $OUTPUT_DIR \
         --do_eval >> "$OUTPUT_DIR/test_log.txt"