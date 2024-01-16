accelerate launch --config_file=./deepspeed_zero3.yaml --gradient_accumulation_steps 16 \
train.py --dataset_name samsum --batch_size 2 --gradient_accumulation_steps 16 \
         --eval_batch_size 8 --output_dir output/mistral-samsum-lora/