accelerate launch --config_file=./deepspeed_zero3.yaml \
train.py --dataset_name summscreen --peft_method lora --num_train_epochs 3 --batch_size 4 \
         --gradient_accumulation_steps 4 --gradient_checkpointing --learning_rate 1e-4 \
         --output_dir output/mistral-summscreen-lora/ \
         --do_train

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name summscreen --peft_method lora \
         --eval_batch_size 1 --output_dir output/mistral-summscreen-lora \
         --do_eval