# export OUTPUT_DIR='output/mistral-samsum-ours-128D'
# mkdir -p $OUTPUT_DIR
# # accelerate launch --config_file=./deepspeed_zero3.yaml \
# # train.py --dataset_name samsum --adapter_method structured --adapter_hidden_size 128 --num_attention_heads 4 \
# #          --injection_location both --num_layers 4 --adapter_type sequential \
# #          --batch_size 4 --gradient_accumulation_steps 4 \
# #          --gradient_checkpointing --learning_rate 1e-4 \
# #          --output_dir $OUTPUT_DIR --do_train >> "$OUTPUT_DIR/train_log.txt"
# accelerate launch --num_processes 4  inference.py --output_dir $OUTPUT_DIR >> "$OUTPUT_DIR/test_log.txt"

# export OUTPUT_DIR='output/mistral-samsum-ours-128D-no-gates' 
# mkdir -p $OUTPUT_DIR
# # accelerate launch --config_file=./deepspeed_zero3.yaml \
# # train.py --dataset_name samsum --adapter_method structured --adapter_hidden_size 128 --num_attention_heads 4 --no_gates\
# #          --injection_location both --num_layers 4 --adapter_type sequential \
# #          --batch_size 4 --gradient_accumulation_steps 4 \
# #          --gradient_checkpointing --learning_rate 1e-4 \
# #          --output_dir $OUTPUT_DIR --do_train >> "$OUTPUT_DIR/train_log.txt"
# accelerate launch --num_processes 4 inference.py --output_dir $OUTPUT_DIR >> "$OUTPUT_DIR/test_log.txt"

export OUTPUT_DIR='output/mistral-samsum-ours-128D'
mkdir -p $OUTPUT_DIR
accelerate launch --config_file=./deepspeed_zero3.yaml \
train.py --dataset_name samsum --use_lora --adapter_method structured --adapter_hidden_size 128 --num_attention_heads 4 \
         --injection_location sa --num_layers 6 --adapter_type parallel \
         --batch_size 4 --gradient_accumulation_steps 4 \
         --gradient_checkpointing --learning_rate 1e-4 \
         --output_dir $OUTPUT_DIR --do_train >> "$OUTPUT_DIR/train_log.txt"
accelerate launch --num_processes 4 inference.py --output_dir $OUTPUT_DIR >> "$OUTPUT_DIR/test_log.txt"

# export OUTPUT_DIR='output/mistral-samsum-our-32D-no-gates'
# mkdir -p $OUTPUT_DIR
# accelerate launch --config_file=./deepspeed_zero3.yaml \
# train.py --dataset_name samsum --adapter_method structured --adapter_hidden_size 32 --num_attention_heads 1 --no_gates \
#          --injection_location both --num_layers 4 --adapter_type sequential \
#          --batch_size 4 --gradient_accumulation_steps 4 \
#          --gradient_checkpointing --learning_rate 1e-4 \
#          --output_dir $OUTPUT_DIR --do_train >> "$OUTPUT_DIR/train_log.txt"
# accelerate launch --num_processes 4 inference.py --output_dir $OUTPUT_DIR >> "$OUTPUT_DIR/test_log.txt"