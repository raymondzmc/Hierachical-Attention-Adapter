# This is a modified version of TRL's `SFTTrainer` example (https://github.com/huggingface/trl/blob/main/examples/scripts/sft_trainer.py), 
# adapted to run with DeepSpeed ZeRO-3 and Mistral-7B-V1.0. The settings below were run on 1 node of 8 x A100 (80GB) GPUs.
#
# Usage:
#   - Install the latest transformers & accelerate versions: `pip install -U transformers accelerate`
#   - Install deepspeed: `pip install deepspeed==0.9.5`
#   - Install TRL from main: pip install git+https://github.com/huggingface/trl.git
#   - Clone the repo: git clone github.com/huggingface/trl.git
#   - Copy this Gist into trl/examples/scripts
#   - Run from root of trl repo with: accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero3.yaml --gradient_accumulation_steps 8 examples/scripts/sft_trainer.py  


import gc
import os
import torch
from typing import Optional, Callable
from dataclasses import dataclass, field
from transformers.modeling_utils import unwrap_model
from accelerate import Accelerator, dispatch_model, infer_auto_device_map, load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    HfArgumentParser,
    Seq2SeqTrainingArguments, 
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    BitsAndBytesConfig,
)
import evaluate

import numpy as np
from seq2seq_trainer import Seq2SeqTrainer
from data import get_preprocessed_samsum, get_preprocessed_summscreen
from datasets import Dataset
import pdb
import nltk

nltk.download('punkt')
tqdm.pandas()

SUPPORTED_DATASETS = ['samsum', 'summscreen']

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """
    seed: Optional[int] = field(default=42)
    model_name: Optional[str] = field(default="mistralai/Mistral-7B-v0.1", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default='samsum', metadata={"help": "the dataset name", "choices": SUPPORTED_DATASETS}
    )
    # do_train: Optional[bool] = field(default=True, metadata={"help": "use 'wandb' to log with wandb"})
    
    report_to: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    optim: Optional[str] = field(default="adamw_torch", metadata={"help": "The optimizer to use"})
    group_by_length: Optional[bool] = field(default=True, metadata={"help": "Whether or not to group together samples of roughly the same length in the training dataset"})
    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default='linear')
    warmup_ratio: Optional[float] = field(default=0.1)
    batch_size: Optional[int] = field(default=2, metadata={"help": "the batch size for training"})
    eval_batch_size: Optional[int] = field(default=2, metadata={"help": "the batch size for evaluation"})
    gradient_checkpointing: Optional[bool] = field(default=False, metadata={"help": "Use gradient checkpointing to save memory at the expense of slower backward pass."})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    logging_steps: Optional[int] = field(default=5, metadata={"help": "the number of logging steps"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    evaluation_strategy: Optional[str] = field(default='epoch', metadata={"help": "The evaluation strategy to adopt during training."})
    save_strategy: Optional[str] = field(default='epoch', metadata={"help": "The checkpoint save strategy to adopt during training."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default="mistral-7b-finetuned-summarization", metadata={"help": "The name of the model on HF Hub"})

def peft_module_casting_to_f16(model):
    from peft.tuners.tuners_utils import BaseTunerLayer

    for name, module in model.named_modules():
        if isinstance(module, BaseTunerLayer):
            module = module.to(torch.float16)
        if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.float16)

def init_trainer(script_args: ScriptArguments,
                 train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Dataset] = None,
                 compute_metrics: Optional[Callable] = None,
                 data_collator: Optional[Callable] = None):
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.float16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )
    if script_args.use_peft:
        model.enable_input_require_grads()
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    generation_config = GenerationConfig(
        max_new_tokens=200,
        do_sample=False,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Step 3: Define the training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        per_device_eval_batch_size=script_args.eval_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=script_args.learning_rate,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_ratio=script_args.warmup_ratio,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        report_to=script_args.report_to,
        save_strategy=script_args.save_strategy,
        evaluation_strategy=script_args.evaluation_strategy,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        fp16=True,
        fp16_full_eval=True,
        logging_first_step=True,
        generation_config=generation_config,
        predict_with_generate=True,
        load_best_model_at_end=True,
    )
    
    # Define the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    return trainer

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Step 1: Load the dataset
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if script_args.dataset_name == 'samsum':
        dataset = get_preprocessed_samsum(tokenizer)
    elif script_args.dataset_name == 'summscreen':
        dataset = get_preprocessed_summscreen(tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer)

    rouge = evaluate.load('rouge')
    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # decode preds and labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # rougeLSum expects newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        return result

    # Initialize the trainer
    trainer = init_trainer(script_args,
                           dataset['train'],
                           dataset['validation'],
                           compute_metrics,
                           data_collator)
    trainer.train()
    trainer.accelerator.wait_for_everyone()

    for checkpoint_dir in os.listdir(script_args.output_dir):
        checkpoint_path = os.path.join(script_args.output_dir, checkpoint_dir)
        if os.path.isdir(checkpoint_path):

            # Initialize a new trainer due to bug in loading checkpoint after `predict`
            del trainer
            torch.cuda.empty_cache()
            gc.collect()
            trainer = init_trainer(script_args, compute_metrics)

            # TODO: This should be refactored into method `predict_from_checkpoint`
            trainer._load_from_checkpoint(checkpoint_path)

            prediction_output = trainer.predict(test_dataset=dataset['test'])
            if trainer.accelerator.is_main_process:
                print(f"Prediction results from checkpoint: {checkpoint_path}:")
                print(prediction_output)
            trainer.accelerator.wait_for_everyone()
