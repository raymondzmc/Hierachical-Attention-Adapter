accelerate launch --config_file=./deepspeed_zero3.yaml \
train.py --dataset_name samsum --num_train_epochs 3 --batch_size 4 --gradient_accumulation_steps 8 \
         --eval_batch_size 1 --gradient_checkpointing --learning_rate 1e-4 \
         --peft_method lora --output_dir output/mistral-samsum-lora \
         --do_eval