# # Parallel to MLP sub-layer (with gates)
# export OUTPUT_DIR='output/mistral-samsum-lora-structured-parallel-mlp-gates'
# # mkdir -p $OUTPUT_DIR
# # accelerate launch --config_file=./deepspeed_zero3.yaml \
# #            train.py --dataset_name samsum \
# #             --use_lora --injection_location mlp --adapter_type parallel --adapter_method structured --pooling_method last \
# #             --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
# #             --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
# #             --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
# #             --do_train >> "$OUTPUT_DIR/train_log.txt"


# # CUDA_VISIBLE_DEVICES=0 python  train.py --dataset_name samsum \
# #          --use_lora --injection_location mlp --adapter_type parallel --adapter_method structured --pooling_method last \
# #          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
# #          --do_eval >> "$OUTPUT_DIR/test_log.txt"


# # # Parallel to MLP sub-layer (no gates)
# export OUTPUT_DIR='output/mistral-samsum-lora-structured-parallel-mlp'
# # mkdir -p $OUTPUT_DIR
# # accelerate launch --config_file=./deepspeed_zero3.yaml \
# #            train.py --dataset_name samsum \
# #             --use_lora --injection_location mlp --adapter_type parallel --no_gates --adapter_method structured --pooling_method last \
# #             --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
# #             --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
# #             --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
# #             --do_train >> "$OUTPUT_DIR/train_log.txt"
# CUDA_VISIBLE_DEVICES=0 python  train.py --dataset_name samsum \
#          --use_lora --injection_location mlp --adapter_type parallel --no_gates --adapter_method structured --pooling_method last \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#          --do_eval 
#         #  >> "$OUTPUT_DIR/test_log.txt"


# # # Sequential to MLP sub-layer
# export OUTPUT_DIR='output/mistral-samsum-lora-structured-sequential-mlp'
# # mkdir -p $OUTPUT_DIR
# # accelerate launch --config_file=./deepspeed_zero3.yaml \
# #            train.py --dataset_name samsum \
# #             --use_lora --injection_location mlp --adapter_type sequential --no_gates --adapter_method structured --pooling_method last \
# #             --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
# #             --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
# #             --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
# #             --do_train >> "$OUTPUT_DIR/train_log.txt"
# CUDA_VISIBLE_DEVICES=0 python  train.py --dataset_name samsum \
#          --use_lora --injection_location mlp --adapter_type sequential --no_gates --adapter_method structured --pooling_method last \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"


# # # Parallel to self-attention sub-layer (with gates)
# export OUTPUT_DIR='output/mistral-samsum-lora-structured-parallel-sa-gates'
# # mkdir -p $OUTPUT_DIR
# # accelerate launch --config_file=./deepspeed_zero3.yaml \
# #            train.py --dataset_name samsum \
# #             --use_lora --injection_location sa --adapter_type parallel --adapter_method structured --pooling_method last \
# #             --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
# #             --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
# #             --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
# #             --do_train >> "$OUTPUT_DIR/train_log.txt"
# CUDA_VISIBLE_DEVICES=0 python  train.py --dataset_name samsum \
#          --use_lora --injection_location sa --adapter_type parallel --adapter_method structured --pooling_method last \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"


# # # Parallel to self-attention sub-layer (no gates)
# export OUTPUT_DIR='output/mistral-samsum-lora-structured-parallel-sa'
# # mkdir -p $OUTPUT_DIR
# # accelerate launch --config_file=./deepspeed_zero3.yaml \
# #            train.py --dataset_name samsum \
# #             --use_lora --injection_location sa --adapter_type parallel --no_gates --adapter_method structured --pooling_method last \
# #             --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
# #             --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
# #             --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
# #             --do_train >> "$OUTPUT_DIR/train_log.txt"
# CUDA_VISIBLE_DEVICES=0 python  train.py --dataset_name samsum \
#          --use_lora --injection_location sa --adapter_type parallel --no_gates --adapter_method structured --pooling_method last \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"


# # # Sequential to self-attention sub-layer
# export OUTPUT_DIR='output/mistral-samsum-lora-structured-sequential-sa'
# # mkdir -p $OUTPUT_DIR
# # accelerate launch --config_file=./deepspeed_zero3.yaml \
# #    train.py --dataset_name samsum \
# #             --use_lora --injection_location sa --adapter_type sequential --no_gates --adapter_method structured --pooling_method last \
# #             --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
# #             --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
# #             --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
# #             --do_train >> "$OUTPUT_DIR/train_log.txt"
# CUDA_VISIBLE_DEVICES=0 python  train.py --dataset_name samsum \
#          --use_lora --injection_location sa --adapter_type sequential --no_gates --adapter_method structured --pooling_method last \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"


# export OUTPUT_DIR='output/mistral-samsum-lora-structured-parallel-mlp-gates-full-attention'
# # mkdir -p $OUTPUT_DIR
# # accelerate launch --config_file=./deepspeed_zero3.yaml \
# #            train.py --dataset_name samsum \
# #                 --use_lora --injection_location mlp --adapter_type parallel --adapter_method structured --pooling_method last \
# #                 --full_attention --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
# #                 --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
# #                 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
# #                 --do_train >> "$OUTPUT_DIR/train_log.txt"
# CUDA_VISIBLE_DEVICES=0 python  train.py --dataset_name samsum \
#          --use_lora --injection_location mlp --adapter_type parallel --adapter_method structured --pooling_method last --full_attention \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"


# export OUTPUT_DIR='output/mistral-samsum-lora-structured-parallel-sa-gates-full-attention'
# mkdir -p $OUTPUT_DIR
# accelerate launch --config_file=./deepspeed_zero3.yaml \
#            train.py --dataset_name samsum \
#             --use_lora --injection_location sa --adapter_type parallel --adapter_method structured --pooling_method last \
#             --full_attention --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
#             --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
#             --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#             --do_train >> "$OUTPUT_DIR/train_log.txt"
# CUDA_VISIBLE_DEVICES=0 python  train.py --dataset_name samsum \
#          --use_lora --injection_location sa --adapter_type parallel --adapter_method structured --pooling_method last --full_attention \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"





# export OUTPUT_DIR='output/mistral-samsum-lora-structured-parallel-sa-gates-no-causal'
# mkdir -p $OUTPUT_DIR
# accelerate launch --config_file=./deepspeed_zero3.yaml \
#            train.py --dataset_name samsum \
#             --use_lora --injection_location sa --adapter_type parallel --adapter_method structured --pooling_method last \
#             --no_causal_attention --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
#             --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
#             --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#             --do_train >> "$OUTPUT_DIR/train_log.txt"
# CUDA_VISIBLE_DEVICES=0 python  train.py --dataset_name samsum \
#          --use_lora --injection_location sa --adapter_type parallel --adapter_method structured --pooling_method last --no_causal_attention \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"

export OUTPUT_DIR='output/mistral-samsum-lora-structured-parallel-mlp-gates-no-causal-4096'
mkdir -p $OUTPUT_DIR
accelerate launch --config_file=./deepspeed_zero3.yaml \
           train.py --dataset_name samsum \
                --use_lora --injection_location mlp --adapter_type parallel --adapter_method structured --pooling_method attention --adapter_hidden_size 4096 --num_attention_heads 32 \
                --no_causal_attention --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
                --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
                --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
                --do_train >> "$OUTPUT_DIR/train_log.txt"


export OUTPUT_DIR='output/mistral-samsum-lora-structured-parallel-mlp-gates-no-causal-last'
mkdir -p $OUTPUT_DIR
accelerate launch --config_file=./deepspeed_zero3.yaml \
           train.py --dataset_name samsum \
                --use_lora --injection_location mlp --adapter_type parallel --adapter_method structured --pooling_method last \
                --no_causal_attention --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
                --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
                --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
                --do_train >> "$OUTPUT_DIR/train_log.txt"
# CUDA_VISIBLE_DEVICES=0 python  train.py --dataset_name samsum \
#          --use_lora --injection_location mlp --adapter_type parallel --adapter_method structured --pooling_method last --no_causal_attention \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"

export OUTPUT_DIR='output/mistral-samsum-lora-structured-parallel-mlp-gates-no-causal-mean'
mkdir -p $OUTPUT_DIR
accelerate launch --config_file=./deepspeed_zero3.yaml \
           train.py --dataset_name samsum \
                --use_lora --injection_location mlp --adapter_type parallel --adapter_method structured --pooling_method mean \
                --no_causal_attention --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
                --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
                --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
                --do_train >> "$OUTPUT_DIR/train_log.txt"
# CUDA_VISIBLE_DEVICES=0 python  train.py --dataset_name samsum \
#          --use_lora --injection_location mlp --adapter_type parallel --adapter_method structured --pooling_method last --no_causal_attention \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"

export OUTPUT_DIR='output/mistral-samsum-lora-structured-parallel-mlp-gates-no-causal-attention'
mkdir -p $OUTPUT_DIR
accelerate launch --config_file=./deepspeed_zero3.yaml \
           train.py --dataset_name samsum \
                --use_lora --injection_location mlp --adapter_type parallel --adapter_method structured --pooling_method attention \
                --no_causal_attention --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
                --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
                --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
                --do_train >> "$OUTPUT_DIR/train_log.txt"
# CUDA_VISIBLE_DEVICES=0 python  train.py --dataset_name samsum \
#          --use_lora --injection_location mlp --adapter_type parallel --adapter_method structured --pooling_method last --no_causal_attention \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"


# export OUTPUT_DIR='output/mistral-samsum-lora-structured-parallel-mlp-gates-no-causal-attention-full'
# mkdir -p $OUTPUT_DIR
# accelerate launch --config_file=./deepspeed_zero3.yaml \
#            train.py --dataset_name samsum \
#                 --use_lora --injection_location mlp --adapter_type parallel --adapter_method structured --pooling_method attention --full_attention \
#                 --no_causal_attention --num_train_epochs 3 --batch_size 16 --gradient_accumulation_steps 1 \
#                 --eval_batch_size 4 --gradient_checkpointing --learning_rate 1e-4 \
#                 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#                 --do_train >> "$OUTPUT_DIR/train_log.txt"
# CUDA_VISIBLE_DEVICES=0 python  train.py --dataset_name samsum \
#          --use_lora --injection_location mlp --adapter_type parallel --adapter_method structured --pooling_method last --no_causal_attention \
#          --eval_batch_size 1 --output_dir $OUTPUT_DIR --adapter_gate_type 'sigmoid' \
#          --do_eval >> "$OUTPUT_DIR/test_log.txt"