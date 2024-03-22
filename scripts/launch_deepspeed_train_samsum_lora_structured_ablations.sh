
# Parallel to MLP sub-layer
export OUTPUT_DIR='output/mistral-samsum-lora-structured-parallel-mlp'
mkdir -p $OUTPUT_DIR
accelerate launch --config_file=./deepspeed_zero3.yaml \
           train.py --dataset_name samsum \
            --injection_location mlp --adapter_type parallel --use_lora --adapter_method structured --pooling_method last \
            --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
            --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
            --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
            --do_train >> "$OUTPUT_DIR/train_log.txt"

# Sequential to MLP sub-layer
export OUTPUT_DIR='output/mistral-samsum-lora-structured-sequential-mlp'
mkdir -p $OUTPUT_DIR
accelerate launch --config_file=./deepspeed_zero3.yaml \
           train.py --dataset_name samsum \
            --injection_location mlp --adapter_type sequential --use_lora --adapter_method structured --pooling_method last \
            --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
            --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
            --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
            --do_train >> "$OUTPUT_DIR/train_log.txt"

# Parallel to self-attention sub-layer
export OUTPUT_DIR='output/mistral-samsum-lora-structured-sequential-mlp'
mkdir -p $OUTPUT_DIR
accelerate launch --config_file=./deepspeed_zero3.yaml \
           train.py --dataset_name samsum \
            --injection_location sa --adapter_type parallel --use_lora --adapter_method structured --pooling_method last \
            --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
            --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
            --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
            --do_train >> "$OUTPUT_DIR/train_log.txt"

# Sequential to self-attention sub-layer
export OUTPUT_DIR='output/mistral-samsum-lora-structured-sequential-mlp'
mkdir -p $OUTPUT_DIR
accelerate launch --config_file=./deepspeed_zero3.yaml \
           train.py --dataset_name samsum \
            --injection_location sa --adapter_type sequential --use_lora --adapter_method structured --pooling_method last \
            --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
            --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
            --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
            --do_train >> "$OUTPUT_DIR/train_log.txt"


# export OUTPUT_DIR='output/mistral-samsum-lora-structured-full-attention'
# mkdir -p $OUTPUT_DIR

# accelerate launch --config_file=./deepspeed_zero3.yaml \
#            train.py --dataset_name samsum --use_lora --adapter_method structured --pooling_method last \
#                 --full_attention --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
#                 --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
#                 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#                 --do_train >> "$OUTPUT_DIR/train_log.txt"