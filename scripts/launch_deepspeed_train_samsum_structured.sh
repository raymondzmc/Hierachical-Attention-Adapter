# export OUTPUT_DIR='output/mistral-samsum-structured-attention-sa'
# mkdir -p $OUTPUT_DIR

# accelerate launch --config_file=./deepspeed_zero3.yaml \
#            train.py --dataset_name samsum --adapter_method structured --pooling_method attention --injection_location attention \
#                 --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
#                 --eval_batch_size 4 --gradient_checkpointing --learning_rate 2e-4 \
#                 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#                 --do_train >> "$OUTPUT_DIR/train_log.txt"

# CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name samsum \
#          --adapter_method structured --pooling_method attention --injection_location attention \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"




# export OUTPUT_DIR='output/mistral-samsum-structured-mean-sa'
# mkdir -p $OUTPUT_DIR

# accelerate launch --config_file=./deepspeed_zero3.yaml \
#            train.py --dataset_name samsum --adapter_method structured --pooling_method mean --injection_location attention \
#                 --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
#                 --eval_batch_size 4 --gradient_checkpointing --learning_rate 2e-4 \
#                 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#                 --do_train >> "$OUTPUT_DIR/train_log.txt"

# CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name samsum \
#          --adapter_method structured --pooling_method mean --injection_location attention \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"




# export OUTPUT_DIR='output/mistral-samsum-structured-last-sa'
# mkdir -p $OUTPUT_DIR

# accelerate launch --config_file=./deepspeed_zero3.yaml \
#            train.py --dataset_name samsum --adapter_method structured --pooling_method last --injection_location attention \
#                 --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
#                 --eval_batch_size 4 --gradient_checkpointing --learning_rate 2e-4 \
#                 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#                 --do_train >> "$OUTPUT_DIR/train_log.txt"

# CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name samsum \
#          --adapter_method structured --pooling_method last --injection_location attention \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"




export OUTPUT_DIR='output/mistral-samsum-structured'
mkdir -p $OUTPUT_DIR

# accelerate launch --config_file=./deepspeed_zero3.yaml train.py --dataset_name samsum \
#          --adapter_method structured --pooling_method last --injection_location sa --adapter_type parallel \
#          --num_layers 32 --use_last_layer --adapter_hidden_size 384 --num_attention_heads 6 \
#          --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
#          --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
#          --output_dir $OUTPUT_DIR --adapter_gate_type 'tanh' \
#          --do_train >> "$OUTPUT_DIR/train_log.txt"

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name samsum \
         --adapter_method structured --pooling_method last --injection_location sa --adapter_type parallel \
         --num_layers 32 --use_last_layer --adapter_hidden_size 384 --num_attention_heads 6 \
         --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'tanh' \
         --do_eval 
        #  >> "$OUTPUT_DIR/test_log.txt"






# export OUTPUT_DIR='output/mistral-samsum-structured-mean-fc'
# mkdir -p $OUTPUT_DIR

# accelerate launch --config_file=./deepspeed_zero3.yaml \
#            train.py --dataset_name samsum --adapter_method structured --pooling_method mean --injection_location fc \
#                 --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
#                 --eval_batch_size 4 --gradient_checkpointing --learning_rate 2e-4 \
#                 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#                 --do_train >> "$OUTPUT_DIR/train_log.txt"

# CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name samsum \
#          --adapter_method structured --pooling_method mean --injection_location fc \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"




# export OUTPUT_DIR='output/mistral-samsum-structured-last-fc'
# mkdir -p $OUTPUT_DIR

# accelerate launch --config_file=./deepspeed_zero3.yaml \
#            train.py --dataset_name samsum --adapter_method structured --pooling_method last --injection_location fc \
#                 --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
#                 --eval_batch_size 4 --gradient_checkpointing --learning_rate 2e-4 \
#                 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#                 --do_train >> "$OUTPUT_DIR/train_log.txt"

# CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name samsum \
#          --adapter_method structured --pooling_method last --injection_location fc \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"