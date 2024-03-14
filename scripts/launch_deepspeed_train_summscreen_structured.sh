export OUTPUT_DIR='output/mistral-summscreen-structured'
mkdir -p $OUTPUT_DIR

# # Debugging only:
# CUDA_VISIBLE_DEVICES=3 python train.py --dataset_name summscreen \
#                        --peft_method structured --num_train_epochs 3 --batch_size 1 \
#                        --gradient_accumulation_steps 8 --gradient_checkpointing --learning_rate 1e-4 \
#                        --output_dir $OUTPUT_DIR --do_train
         

# accelerate launch --config_file=./deepspeed_zero3.yaml \
# train.py --dataset_name summscreen --peft_method structured \
#          --num_train_epochs 3 --batch_size 2 --gradient_accumulation_steps 4 --learning_rate 1e-4 \
#          --gradient_checkpointing --output_dir $OUTPUT_DIR \
#          --do_train >> "$OUTPUT_DIR/ds_train_log.txt"


CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name summscreen --peft_method structured  \
                                       --eval_batch_size 1 --output_dir $OUTPUT_DIR \
                                       --do_eval --checkpoint_dir checkpoint-1773 >> "$OUTPUT_DIR/test_log_1773.txt"

CUDA_VISIBLE_DEVICES=1 python train.py --dataset_name summscreen --peft_method structured  \
                                       --eval_batch_size 1 --output_dir $OUTPUT_DIR \
                                       --do_eval --checkpoint_dir checkpoint-1182 >> "$OUTPUT_DIR/test_log_1182.txt"

