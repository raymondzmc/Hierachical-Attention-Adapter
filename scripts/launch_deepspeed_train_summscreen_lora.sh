export OUTPUT_DIR='output/mistral-summscreen-lora'
mkdir -p $OUTPUT_DIR

# accelerate launch --config_file=./deepspeed_zero3.yaml \
# train.py --dataset_name summscreen --peft_method lora --num_train_epochs 3 --batch_size 4 \
#          --gradient_accumulation_steps 4 --gradient_checkpointing --learning_rate 1e-4 \
#          --output_dir $OUTPUT_DIR \
#          --do_train >> "$OUTPUT_DIR/ds_train_log.txt"

CUDA_VISIBLE_DEVICES=2 python train.py --dataset_name summscreen --peft_method lora \
         --eval_batch_size 1 --output_dir $OUTPUT_DIR \
         --do_eval >> "$OUTPUT_DIR/test_log.txt"