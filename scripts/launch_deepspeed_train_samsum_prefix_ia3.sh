export OUTPUT_DIR='output/mistral-samsum-prefix'
mkdir -p $OUTPUT_DIR

# Debugging only:
# CUDA_VISIBLE_DEVICES=0 python train.py \
#          --dataset_name samsum  --use_prefix \
#          --num_train_epochs 3 --batch_size 4 --gradient_accumulation_steps 4 \
#          --gradient_checkpointing --learning_rate 2e-5 \
#          --output_dir $OUTPUT_DIR --do_train

accelerate launch --config_file=./deepspeed_zero3.yaml \
train.py --dataset_name samsum --use_prefix \
         --num_train_epochs 3 --batch_size 4 --gradient_accumulation_steps 4 \
         --eval_batch_size 4 --gradient_checkpointing --learning_rate 2e-5 \
         --output_dir $OUTPUT_DIR --do_train >> "$OUTPUT_DIR/train_log.txt"

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name samsum --use_prefix \
         --eval_batch_size 1 --output_dir $OUTPUT_DIR \
         --do_eval >> "$OUTPUT_DIR/test_log.txt"


export OUTPUT_DIR='output/mistral-samsum-ia3'
mkdir -p $OUTPUT_DIR

# # Debugging only:
# CUDA_VISIBLE_DEVICES=0 python train.py \
#          --dataset_name samsum  --use_ia3 \
#          --num_train_epochs 3 --batch_size 4 --gradient_accumulation_steps 4 \
#          --gradient_checkpointing --learning_rate 2e-5 \
#          --output_dir $OUTPUT_DIR --do_train

accelerate launch --config_file=./deepspeed_zero3.yaml \
train.py --dataset_name samsum --use_ia3 \
         --num_train_epochs 3 --batch_size 4 --gradient_accumulation_steps 4 \
         --eval_batch_size 4 --gradient_checkpointing --learning_rate 2e-5 \
         --output_dir $OUTPUT_DIR --do_train >> "$OUTPUT_DIR/train_log.txt"

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name samsum --use_ia3 \
         --eval_batch_size 1 --output_dir $OUTPUT_DIR \
         --do_eval >> "$OUTPUT_DIR/test_log.txt"