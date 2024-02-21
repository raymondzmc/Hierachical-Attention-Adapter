export OUTPUT_DIR='output/mistral-samsum-lora-structured-sigmoid'
mkdir -p $OUTPUT_DIR

# Debugging only:
# CUDA_VISIBLE_DEVICES=0 python train.py \
#          --dataset_name samsum  --use_lora --adapter_method structured  \
#          --num_train_epochs 3 --batch_size 4 --gradient_accumulation_steps 4 \
#          --gradient_checkpointing --learning_rate 2e-5 --adapter_gate_type 'sigmoid' \
#          --output_dir $OUTPUT_DIR --do_train

accelerate launch --config_file=./deepspeed_zero3.yaml \
train.py --dataset_name samsum --use_lora --adapter_method structured \
         --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
         --eval_batch_size 4 --gradient_checkpointing --learning_rate 2e-5 \
         --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
         --do_train >> "$OUTPUT_DIR/train_log.txt"

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name samsum \
         --use_lora --adapter_method structured \
         --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
         --do_eval >> "$OUTPUT_DIR/test_log.txt"


export OUTPUT_DIR='output/mistral-samsum-lora-structured-tanh'
mkdir -p $OUTPUT_DIR

accelerate launch --config_file=./deepspeed_zero3.yaml \
train.py --dataset_name samsum --use_lora --adapter_method structured \
         --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
         --eval_batch_size 4 --gradient_checkpointing --learning_rate 2e-5 \
         --output_dir $OUTPUT_DIR --adapter_gate_type 'tanh' \
         --do_train >> "$OUTPUT_DIR/train_log.txt"

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name samsum \
         --use_lora --adapter_method structured \
         --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'tanh' \
         --do_eval >> "$OUTPUT_DIR/test_log.txt"