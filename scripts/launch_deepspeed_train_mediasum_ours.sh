export OUTPUT_DIR='output/mistral-mediasum-lora-structured'
mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29501 --config_file=./deepspeed_zero3.yaml \
train.py --dataset_name mediasum \
         --use_lora --adapter_method structured --pooling_method attention --injection_location fc --adapter_gate_type sigmoid \
         --num_train_epochs 1 --batch_size 4 \
         --save_strategy steps --save_steps 0.25 --evaluation_strategy no \
         --gradient_accumulation_steps 4 --gradient_checkpointing --learning_rate 1e-4 \
         --output_dir $OUTPUT_DIR \
         --do_train >> "$OUTPUT_DIR/ds_train_log.txt"

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name mediasum  \
         --use_lora --adapter_method structured --pooling_method attention --injection_location fc --adapter_gate_type sigmoid \
         --eval_batch_size 1 --output_dir $OUTPUT_DIR \
         --do_eval --test_subset 1000 >> "$OUTPUT_DIR/test_log.txt"


export OUTPUT_DIR='output/mistral-mediasum-structured'
mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29501 --config_file=./deepspeed_zero3.yaml \
train.py --dataset_name mediasum \
         --adapter_method structured --pooling_method attention --injection_location fc --adapter_gate_type sigmoid \
         --num_train_epochs 1 --batch_size 4 \
         --save_strategy steps --save_steps 0.25 --evaluation_strategy no \
         --gradient_accumulation_steps 4 --gradient_checkpointing --learning_rate 1e-4 \
         --output_dir $OUTPUT_DIR \
         --do_train >> "$OUTPUT_DIR/ds_train_log.txt"

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name mediasum  \
         --adapter_method structured --pooling_method attention --injection_location fc --adapter_gate_type sigmoid \
         --eval_batch_size 1 --output_dir $OUTPUT_DIR \
         --do_eval --test_subset 1000 >> "$OUTPUT_DIR/test_log.txt"