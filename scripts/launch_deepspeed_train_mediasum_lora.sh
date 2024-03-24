export OUTPUT_DIR='output/mistral-mediasum-lora'
mkdir -p $OUTPUT_DIR

# Debugging only:
# CUDA_VISIBLE_DEVICES=0 python train.py \
#          --dataset_name mediasum  --use_lora \
#          --num_train_epochs 1 --batch_size 4 --gradient_accumulation_steps 4 \
#          --gradient_checkpointing --learning_rate 2e-5 \
#          --output_dir $OUTPUT_DIR --do_train

accelerate launch --config_file=./deepspeed_zero3.yaml \
train.py --dataset_name mediasum --use_lora --num_train_epochs 1 --batch_size 1 \
         --save_strategy steps --save_steps 0.25 --evaluation_strategy no \
         --gradient_accumulation_steps 16 --gradient_checkpointing --learning_rate 1e-4 \
         --output_dir $OUTPUT_DIR \
         --do_train >> "$OUTPUT_DIR/ds_train_log.txt"

# CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name mediasum  \
#          --use_lora --eval_batch_size 1 --output_dir $OUTPUT_DIR \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"