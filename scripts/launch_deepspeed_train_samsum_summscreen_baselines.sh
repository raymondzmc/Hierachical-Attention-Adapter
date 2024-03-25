# export OUTPUT_DIR='output/mistral-samsum-lora'
# mkdir -p $OUTPUT_DIR
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file=./deepspeed_zero3.yaml \
#  train.py --dataset_name samsum  --use_lora \
#          --num_train_epochs 3 --batch_size 4 --gradient_accumulation_steps 4 \
#          --gradient_checkpointing --learning_rate 1e-4 \
#          --output_dir $OUTPUT_DIR --do_train >> "$OUTPUT_DIR/train_log.txt"
# CUDA_VISIBLE_DEVICES=3 python  train.py --dataset_name samsum \
#          --use_lora \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"


# export OUTPUT_DIR='output/mistral-samsum-ia3'
# mkdir -p $OUTPUT_DIR
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file=./deepspeed_zero3.yaml \
# train.py --dataset_name samsum --use_ia3 \
#          --num_train_epochs 3 --batch_size 4 --gradient_accumulation_steps 4 \
#          --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
#          --output_dir $OUTPUT_DIR --do_train >> "$OUTPUT_DIR/train_log.txt"
# CUDA_VISIBLE_DEVICES=3 python  train.py --dataset_name samsum \
#          --use_ia3 \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"

# export OUTPUT_DIR='output/mistral-samsum-adapter'
# mkdir -p $OUTPUT_DIR
# accelerate launch --config_file=./deepspeed_zero3.yaml \
# train.py --dataset_name samsum --adapter_method mlp --injection_location both --num_layers 32 --use_last_layer --adapter_type sequential \
#          --num_train_epochs 3 --batch_size 4 --gradient_accumulation_steps 4 \
#          --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
#          --output_dir $OUTPUT_DIR --do_train >> "$OUTPUT_DIR/train_log.txt"


export OUTPUT_DIR='output/mistral-summscreen-lora'
mkdir -p $OUTPUT_DIR
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file=./deepspeed_zero3.yaml \
#  train.py --dataset_name summscreen  --use_lora \
#         --num_train_epochs 3 --batch_size 4 --gradient_accumulation_steps 4 \
#         --gradient_checkpointing --learning_rate 1e-4 \
#         --output_dir $OUTPUT_DIR --do_train >> "$OUTPUT_DIR/train_log.txt"
CUDA_VISIBLE_DEVICES=0 python  train.py --dataset_name summscreen \
        --use_lora \
        --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
        --do_eval --test_subset 200
        # >> "$OUTPUT_DIR/test_log.txt"



# export OUTPUT_DIR='output/mistral-summscreen-ia3'
# mkdir -p $OUTPUT_DIR
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file=./deepspeed_zero3.yaml \
# train.py --dataset_name summscreen --use_ia3 \
#          --num_train_epochs 3 --batch_size 4 --gradient_accumulation_steps 4 \
#          --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
#          --output_dir $OUTPUT_DIR --do_train >> "$OUTPUT_DIR/train_log.txt"