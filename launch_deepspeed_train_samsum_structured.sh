accelerate launch --config_file=./deepspeed_zero3.yaml \
train.py --dataset_name samsum --num_train_epochs 3 --batch_size 4 --gradient_accumulation_steps 2 \
         --eval_batch_size 1 --gradient_checkpointing --learning_rate 1e-4 \
         --peft_method structured --output_dir output/mistral-samsum-structured \
         --do_train

CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name samsum \
         --eval_batch_size 1  --peft_method attention --output_dir output/mistral-samsum-attention \
         --do_eval