# export OUTPUT_DIR='output/mistral-summscreen-lora'
# mkdir -p $OUTPUT_DIR

export OUTPUT_DIR='output/mistral-summscreen-lora-r-32-alpha-8'
mkdir -p $OUTPUT_DIR
accelerate launch --config_file=./deepspeed_zero3.yaml \
train.py --dataset_name summscreen --use_lora --lora_r 32 --lora_alpha 8  \
         --num_train_epochs 3 --batch_size 2 --gradient_accumulation_steps 8 \
         --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
         --output_dir $OUTPUT_DIR --do_train >> "$OUTPUT_DIR/train_log.txt"
accelerate launch --num_processes 4  inference.py --output_dir $OUTPUT_DIR

# CUDA_VISIBLE_DEVICES=2 python train.py --dataset_name summscreen --peft_method lora \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"