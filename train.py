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


import os
import json
import torch
from tqdm import tqdm
from transformers import HfArgumentParser, AutoTokenizer
from mistral import init_mistral_model
from utils import ScriptArguments, init_seq2seq_trainer, StopSequenceRepeatCriteria
from data import DataCollatorForSeq2Seq, get_dataset_and_metrics

tqdm.pandas()


def peft_module_casting_to_f16(model):
    from peft.tuners.tuners_utils import BaseTunerLayer

    for name, module in model.named_modules():
        if isinstance(module, BaseTunerLayer):
            module = module.to(torch.float16)
        if 'adapter' in name:
            if hasattr(module, "weight"):
                module = module.to(torch.float16)
        if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.float16)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    if script_args.config_file and os.path.isfile(script_args.config_file):
        script_args = parser.parse_json_file(os.path.join(script_args.output_dir, 'config.json'))
    else:
        # Write config to output directory
        with open(os.path.join(script_args.output_dir, 'config.json'), 'w') as f:
            json.dump(script_args.to_dict(), f)

    # Environment variables for Weights & Biases
    if script_args.report_to == 'wandb':
        os.environ["WANDB_PROJECT"] = "Dialogue Summarization"
        os.environ["WANDB_WATCH"] = "false"
        os.environ["WANDB_NAME"] = os.path.basename(script_args.output_dir)
    
    # Step 1: Load pre-trained model and tokenizer
    model = init_mistral_model(script_args)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, padding_side="left")
    
    # Step 2: Load the dataset
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return_token_type_ids = (script_args.adapter_method == 'structured')
    dataset, compute_metrics = get_dataset_and_metrics(script_args.dataset_name, tokenizer, return_token_type_ids)
    data_collator = DataCollatorForSeq2Seq(tokenizer)
    stopping_criteria = StopSequenceRepeatCriteria(tokenizer=tokenizer,
                                                   stop_sequences=['\n', '\r', '  '],
                                                   max_repeat=4)

    # Initialize the trainer
    trainer = init_seq2seq_trainer(script_args,
                                   model,
                                   tokenizer,
                                   dataset['train'],
                                   dataset['validation'],
                                   compute_metrics,
                                   data_collator)
    trainer.train()
    trainer.accelerator.wait_for_everyone()
