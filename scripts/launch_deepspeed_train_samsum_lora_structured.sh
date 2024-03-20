# export OUTPUT_DIR='output/mistral-samsum-lora-structured-attention-sa'
# mkdir -p $OUTPUT_DIR

# accelerate launch --config_file=./deepspeed_zero3.yaml \
#            train.py --dataset_name samsum --use_lora --adapter_method structured --pooling_method attention --injection_location attention \
#                 --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
#                 --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
#                 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#                 --do_train >> "$OUTPUT_DIR/train_log.txt"

# CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name samsum \
#          --use_lora --adapter_method structured --pooling_method attention --injection_location attention \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"




# export OUTPUT_DIR='output/mistral-samsum-lora-structured-mean-sa'
# mkdir -p $OUTPUT_DIR

# accelerate launch --config_file=./deepspeed_zero3.yaml \
#            train.py --dataset_name samsum --use_lora --adapter_method structured --pooling_method mean --injection_location attention \
#                 --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
#                 --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
#                 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#                 --do_train >> "$OUTPUT_DIR/train_log.txt"

# CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name samsum \
#          --use_lora --adapter_method structured --pooling_method mean --injection_location attention \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"




# export OUTPUT_DIR='output/mistral-samsum-lora-structured-last-sa'
# mkdir -p $OUTPUT_DIR

# accelerate launch --config_file=./deepspeed_zero3.yaml \
#            train.py --dataset_name samsum --use_lora --adapter_method structured --pooling_method last --injection_location attention \
#                 --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
#                 --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
#                 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#                 --do_train >> "$OUTPUT_DIR/train_log.txt"

# CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name samsum \
#          --use_lora --adapter_method structured --pooling_method last --injection_location attention \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"




# export OUTPUT_DIR='output/mistral-samsum-lora-structured-attention-fc'
# mkdir -p $OUTPUT_DIR

# accelerate launch --config_file=./deepspeed_zero3.yaml \
#            train.py --dataset_name samsum --use_lora --adapter_method structured --pooling_method attention --injection_location fc \
#                 --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
#                 --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
#                 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#                 --do_train >> "$OUTPUT_DIR/train_log.txt"

# CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name samsum \
#          --use_lora --adapter_method structured --pooling_method attention --injection_location fc \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"






export OUTPUT_DIR='output/mistral-samsum-structured-mean-fc'
mkdir -p $OUTPUT_DIR

# accelerate launch --config_file=./deepspeed_zero3.yaml \
#            train.py --dataset_name samsum --use_lora --adapter_method structured --pooling_method mean --injection_location fc \
#                 --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
#                 --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
#                 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#                 --do_train >> "$OUTPUT_DIR/train_log.txt"

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name samsum \
         --use_lora --adapter_method structured --pooling_method mean --injection_location fc \
         --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
         --do_eval >> "$OUTPUT_DIR/test_log.txt"




# export OUTPUT_DIR='output/mistral-samsum-lora-structured-last-fc'
# mkdir -p $OUTPUT_DIR

# accelerate launch --config_file=./deepspeed_zero3.yaml \
#            train.py --dataset_name samsum --use_lora --adapter_method structured --pooling_method last --injection_location fc \
#                 --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
#                 --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
#                 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#                 --do_train >> "$OUTPUT_DIR/train_log.txt"

# CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name samsum \
#          --use_lora --adapter_method structured --pooling_method last --injection_location fc \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"