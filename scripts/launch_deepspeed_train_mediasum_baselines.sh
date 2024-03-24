export OUTPUT_DIR='output/mistral-mediasum-lora'
mkdir -p $OUTPUT_DIR
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file=./deepspeed_zero3.yaml \
 train.py --dataset_name mediasum  --use_lora \
         --num_train_epochs 1 --evaluation_strategy steps --save_strategy steps --batch_size 4 --gradient_accumulation_steps 4 \
         --gradient_checkpointing --learning_rate 1e-4 \
         --output_dir $OUTPUT_DIR --do_train >> "$OUTPUT_DIR/train_log.txt"


export OUTPUT_DIR='output/mistral-mediasum-ia3'
mkdir -p $OUTPUT_DIR
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file=./deepspeed_zero3.yaml \
train.py --dataset_name mediasum --use_ia3 \
         --num_train_epochs 1 --evaluation_strategy steps --save_strategy steps --batch_size 4 --gradient_accumulation_steps 4 \
         --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
         --output_dir $OUTPUT_DIR --do_train >> "$OUTPUT_DIR/train_log.txt"

# export OUTPUT_DIR='output/mistral-samsum-adapter'
# mkdir -p $OUTPUT_DIR
# accelerate launch --config_file=./deepspeed_zero3.yaml \
# python train.py --dataset_name samsum --adapter_method mlp --injection_location both --num_layers 32 --use_last_layer --adapter_type sequential \
#          --num_train_epochs 3 --batch_size 4 --gradient_accumulation_steps 4 \
#          --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
#          --output_dir $OUTPUT_DIR --do_train >> "$OUTPUT_DIR/train_log.txt"
