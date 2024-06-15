import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from accelerate import PartialState
from accelerate.utils import gather_object, SAFE_WEIGHTS_INDEX_NAME
from transformers import HfArgumentParser, AutoTokenizer, AutoModel
from transformers.modeling_utils import load_sharded_checkpoint
import safetensors.torch as st_torch
from peft import PeftModel

from data import get_dataset_and_metrics
from data.metrics import rouge_metrics, gpt_score_metrics
from utils import ScriptArguments
from mistral import init_mistral_model
import pdb

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    random.seed(script_args.seed)
    np.random.seed(script_args.seed)

    if script_args.output_dir is None:
        raise Exception("Argument `--output_dir` needs to be specified!")
    elif not os.path.exists(script_args.output_dir):
        raise FileNotFoundError(f"Output directory `{script_args.output_dir}` doesn't exists")
    
    if script_args.config_file is not None:
        config_file = script_args.config_file
    else:
        config_file = os.path.join(script_args.output_dir, 'config.json')

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Cannot find configuration file at {config_file}")

    # Get script arguments from config file
    script_args.update(parser.parse_json_file(config_file)[0])
    
    # Initialize model and tokenizer
    model = init_mistral_model(script_args)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, padding_side="left")

    # Get test dataset and compute metrics
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return_token_type_ids = (script_args.adapter_method == 'structured')
    test_dataset, compute_metrics = get_dataset_and_metrics(
        script_args.dataset_name, tokenizer, return_token_type_ids=return_token_type_ids, test_only=True)
    
    # Get checkpoint directories
    checkpoint_paths = [os.path.join(script_args.output_dir, dir) for dir in os.listdir(script_args.output_dir)]
    checkpoint_paths = [path for path in checkpoint_paths if os.path.isdir(path)]
    checkpoint_paths = sorted(checkpoint_paths, key=lambda x: int(x.split('-')[-1]), reverse=True)
    
    distributed_state = PartialState()
    device = distributed_state.device
    for checkpoint_path in checkpoint_paths:
        RESULT_FILE = os.path.join(checkpoint_path, 'results.json')
        
        if not os.path.exists(RESULT_FILE):
            
            # Load fine-tuned adapter weights
            if isinstance(model, PeftModel):
                model.load_adapter(checkpoint_path, model.active_adapter)
            elif os.path.exists(os.path.join(checkpoint_path, SAFE_WEIGHTS_INDEX_NAME)):
                load_sharded_checkpoint(model, checkpoint_path)
            else:
                raise NotImplementedError("Need to implement loading for other save types")

            model = model.to(device)
            results = []
            with distributed_state.split_between_processes(list(range(len(test_dataset)))) as dataset_indices:
                slit_dataset = test_dataset.select(dataset_indices)
                for example in tqdm(slit_dataset):
                    labels = torch.tensor(example['labels'])
                    input_ids = torch.tensor(example['input_ids'])[labels == -100]
                    label_ids = torch.tensor(example['input_ids'])[labels != -100]
                    attention_mask = torch.tensor(example['attention_mask'])[:len(input_ids)]
                    token_type_ids = torch.tensor(example['token_type_ids'])[:len(input_ids)]
                    output = model.generate(input_ids=input_ids.to(device).unsqueeze(0),
                                            attention_mask=attention_mask.to(device).unsqueeze(0),
                                            max_new_tokens=500,
                                            do_sample=False,
                                            pad_token_id=tokenizer.eos_token_id,
                                            token_type_ids=token_type_ids.to(device).unsqueeze(0) if return_token_type_ids else None)
                    pred_ids = output.squeeze(0)[len(input_ids):]
                    ref_text = tokenizer.decode(label_ids, skip_special_tokens=True)
                    pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
                    results.append({
                        'id': example['id'],
                        # 'dialogue': example['dialogue'],
                        'reference': ref_text,
                        'prediction': pred_text,
                    })

            # Remove duplicated examples from gathered results
            results = list({x['id']: x for x in gather_object(results)}.values())
            with open(RESULT_FILE, 'w') as f:
                json.dump(results, f, indent=4)

        else:
            # Load pre-computed inference results
            results = json.load(open(RESULT_FILE))
        
        
        if distributed_state.is_main_process:
            # Compute ROUGE and GPT score and prepend them to results
            references = [x['reference'] for x in results if 'reference' in x]
            predictions = [x['prediction'] for x in results if 'prediction' in x]
            assert len(references) == len(predictions), \
                "Number of references does not match the number of predictions!"
            
            print(f"\nEvaluation Results from `{checkpoint_path}`")
            rouge_scores = rouge_metrics(references, predictions)
            results.insert(0, rouge_scores)
            print(f"[ROUGE]: {rouge_scores}")
            
            # gpt3_scores = gpt_score_metrics(references, predictions)
            # results.insert(0, gpt3_scores)
            # print(f"[GPT3Score]: {gpt3_scores}")
            
            with open(RESULT_FILE, 'w') as f:
                json.dump(results, f, indent=4)
        
