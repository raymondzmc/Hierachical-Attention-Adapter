
# export OUTPUT_DIR='output/mistral-samsum-attention-32D'
# mkdir -p $OUTPUT_DIR
# accelerate launch --config_file=./deepspeed_zero3.yaml \
# train.py --dataset_name samsum --adapter_method attention --adapter_hidden_size 32 --num_attention_heads 1 \
#          --injection_location both --num_layers 31 --adapter_type sequential \
#          --batch_size 4 --gradient_accumulation_steps 4 \
#          --gradient_checkpointing --learning_rate 1e-4 \
#          --output_dir $OUTPUT_DIR --do_train >> "$OUTPUT_DIR/train_log.txt"
# accelerate launch --num_processes 4 inference.py --output_dir $OUTPUT_DIR >> "$OUTPUT_DIR/test_log.txt"

# export OUTPUT_DIR='output/mistral-samsum-attention-64D'
# mkdir -p $OUTPUT_DIR
# accelerate launch --config_file=./deepspeed_zero3.yaml \
# train.py --dataset_name samsum --adapter_method attention --adapter_hidden_size 64 --num_attention_heads 1 \
#          --injection_location both --num_layers 31 --adapter_type sequential \
#          --batch_size 4 --gradient_accumulation_steps 4 \
#          --gradient_checkpointing --learning_rate 1e-4 \
#          --output_dir $OUTPUT_DIR --do_train >> "$OUTPUT_DIR/train_log.txt"
# accelerate launch --num_processes 4 inference.py --output_dir $OUTPUT_DIR >> "$OUTPUT_DIR/test_log.txt"

# export OUTPUT_DIR='output/mistral-samsum-attention-32D'
# mkdir -p $OUTPUT_DIR
# accelerate launch --config_file=./deepspeed_zero3.yaml \
# train.py --dataset_name samsum --adapter_method attention --adapter_hidden_size 64 --num_attention_heads 1 \
#          --injection_location both --num_layers 31 --adapter_type sequential \
#          --batch_size 4 --gradient_accumulation_steps 4 \
#          --gradient_checkpointing --learning_rate 1e-4 \
#          --output_dir $OUTPUT_DIR --do_train >> "$OUTPUT_DIR/train_log.txt"
# accelerate launch --num_processes 4 inference.py --output_dir $OUTPUT_DIR >> "$OUTPUT_DIR/test_log.txt"

# export OUTPUT_DIR='output/mistral-samsum-attention-64D'
# mkdir -p $OUTPUT_DIR
# accelerate launch --config_file=./deepspeed_zero3.yaml \
# train.py --dataset_name samsum --adapter_method attention --adapter_hidden_size 64 --num_attention_heads 1 \
#          --injection_location both --num_layers 31 --adapter_type sequential \
#          --batch_size 4 --gradient_accumulation_steps 4 \
#          --gradient_checkpointing --learning_rate 1e-4 \
#          --output_dir $OUTPUT_DIR --do_train >> "$OUTPUT_DIR/train_log.txt"
# accelerate launch --num_processes 4 inference.py --output_dir $OUTPUT_DIR >> "$OUTPUT_DIR/test_log.txt"


# export OUTPUT_DIR='output/mistral-samsum-attention-32D'
# mkdir -p $OUTPUT_DIR
# accelerate launch --config_file=./deepspeed_zero3.yaml \
# train.py --dataset_name samsum --adapter_method attention --adapter_hidden_size 16 --num_attention_heads 1 \
#          --injection_location both --num_layers 32 --use_last_layer --adapter_type sequential \
#          --batch_size 4 --gradient_accumulation_steps 4 \
#          --gradient_checkpointing --learning_rate 1e-4 \
#          --output_dir $OUTPUT_DIR --do_train >> "$OUTPUT_DIR/train_log.txt"
# # accelerate launch --num_processes 4 inference.py --output_dir $OUTPUT_DIR >> "$OUTPUT_DIR/test_log.txt"

# export OUTPUT_DIR='output/mistral-samsum-attention-64D'
# mkdir -p $OUTPUT_DIR
# accelerate launch --config_file=./deepspeed_zero3.yaml \
# train.py --dataset_name samsum --adapter_method attention --adapter_hidden_size 16 --num_attention_heads 1 \
#          --injection_location both --num_layers 32 --use_last_layer --adapter_type sequential \
#          --batch_size 4 --gradient_accumulation_steps 4 \
#          --gradient_checkpointing --learning_rate 1e-4 \
#          --output_dir $OUTPUT_DIR --do_train >> "$OUTPUT_DIR/train_log.txt"
# accelerate launch --num_processes 4 inference.py --output_dir $OUTPUT_DIR >> "$OUTPUT_DIR/test_log.txt"
