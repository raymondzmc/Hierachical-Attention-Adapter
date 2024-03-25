export OUTPUT_DIR='output/mistral-summscreen-lora-structured'
mkdir -p $OUTPUT_DIR

accelerate launch --config_file=./deepspeed_zero3.yaml \
train.py --dataset_name summscreen \
         --use_lora --adapter_method structured --pooling_method attention --injection_location fc --adapter_gate_type sigmoid \
         --num_train_epochs 3 --batch_size 2 --gradient_accumulation_steps 8 \
         --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
         --output_dir $OUTPUT_DIR \
         --do_train >> "$OUTPUT_DIR/train_log.txt"

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name summscreen \
         --use_lora --adapter_method structured --pooling_method attention --injection_location fc --adapter_gate_type sigmoid \
         --eval_batch_size 1 --output_dir $OUTPUT_DIR \
         --do_eval --test_subset 300 >> "$OUTPUT_DIR/test_log.txt"

export OUTPUT_DIR='output/mistral-summscreen-structured'
mkdir -p $OUTPUT_DIR
accelerate launch --config_file=./deepspeed_zero3.yaml \
train.py --dataset_name summscreen \
         --adapter_method structured --pooling_method attention --injection_location fc --adapter_gate_type sigmoid \
         --num_train_epochs 3 --batch_size 1 --gradient_accumulation_steps 16 \
         --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
         --output_dir $OUTPUT_DIR \
         --do_train >> "$OUTPUT_DIR/train_log.txt"

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name summscreen \
         --adapter_method structured --pooling_method attention --injection_location fc --adapter_gate_type sigmoid \
         --eval_batch_size 1 --output_dir $OUTPUT_DIR \
         --do_eval --test_subset 300 >> "$OUTPUT_DIR/test_log.txt"