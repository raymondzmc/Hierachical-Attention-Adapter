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
import json
import torch
from typing import Optional, Callable
from dataclasses import dataclass, field
from transformers.modeling_utils import unwrap_model
from accelerate import Accelerator, dispatch_model, infer_auto_device_map, load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory
from peft import LoraConfig, PrefixTuningConfig, IA3Config, PromptTuningConfig, PromptEncoderConfig, TaskType, get_peft_model
from tqdm import tqdm
from mistral import MistralForCausalLM, StructuredAdapterConfig, AttentionAdapterConfig, MLPAdapterConfig
from transformers import (
    AutoModelForCausalLM, 
    HfArgumentParser,
    Seq2SeqTrainingArguments, 
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
    StoppingCriteriaList,
    StoppingCriteria,
)
from rouge_score import rouge_scorer, scoring
from metrics import f1_score, exact_match_score, metric_max_over_ground_truths

import numpy as np
from seq2seq_trainer import Seq2SeqTrainer
from data import (
    get_preprocessed_samsum,
    get_preprocessed_summscreen, 
    get_preprocessed_mediasum,
    get_preprocessed_friendsqa,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
import pdb
import nltk
from torch.utils.data import Subset
tqdm.pandas()

SUPPORTED_DATASETS = ['samsum', 'summscreen', 'mediasum', 'friendsqa']
SUPPORTED_ADAPTER_METHODS = ['attention', 'structured', 'mlp']

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
    do_train: Optional[bool] = field(default=False, metadata={"help": "Whether to perform training"})
    do_eval: Optional[bool] = field(default=False, metadata={"help": "Whether to perform evaluation"})
    
    report_to: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    optim: Optional[str] = field(default="adamw_torch", metadata={"help": "The optimizer to use"})
    group_by_length: Optional[bool] = field(default=True, metadata={"help": "Whether or not to group together samples of roughly the same length in the training dataset"})
    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default='constant_with_warmup')
    warmup_steps: Optional[float] = field(default=20)
    batch_size: Optional[int] = field(default=2, metadata={"help": "the batch size for training"})
    eval_batch_size: Optional[int] = field(default=2, metadata={"help": "the batch size for evaluation"})
    gradient_checkpointing: Optional[bool] = field(default=False, metadata={"help": "Use gradient checkpointing to save memory at the expense of slower backward pass."})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_lora: Optional[bool] = field(default=False, metadata={"help": "Whether to add LoRA"})
    use_prefix: Optional[bool] = field(default=False, metadata={"help": "Whether to add Prefix Tuning"})
    use_ia3: Optional[bool] = field(default=False, metadata={"help": "Whether to add IA3"})
    use_ptuning: Optional[bool] = field(default=False, metadata={"help": "Whether to add P-Tuning"})
    adapter_method: Optional[str] = field(default='lora', metadata={"help": "Adapter method to use", "choices": SUPPORTED_ADAPTER_METHODS})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    checkpoint_dir: Optional[str] = field(default=None, metadata={"help": "the checkpoint directory for evaluation"})
    logging_steps: Optional[int] = field(default=5, metadata={"help": "the number of logging steps"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    evaluation_strategy: Optional[str] = field(default='epoch', metadata={"help": "The evaluation strategy to adopt during training."})
    eval_steps: Optional[float] = field(default=0.25)
    save_strategy: Optional[str] = field(default='epoch', metadata={"help": "The checkpoint save strategy to adopt during training."})
    save_steps: Optional[float] = field(default=0.25)
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default="mistral-7b-finetuned-summarization", metadata={"help": "The name of the model on HF Hub"})
    resume_from_checkpoint: Optional[str] = field(default=None, metadata={"help": "The path to folder with a valid checkpoint to load from."})
    
    # Our defined arguments
    adapter_gate_type: Optional[str] = field(default='sigmoid', metadata={"choices": ['sigmoid', 'tanh']})
    pooling_method: Optional[str] = field(default='mean', metadata={"choices": ['mean', 'attention', 'last']})
    injection_location: Optional[str] = field(default='attention', metadata={"choices": ['sa', 'mlp', 'both']})
    adapter_type: Optional[str] = field(default='parallel', metadata={"choices": ['parallel', 'sequential']})
    full_attention: Optional[bool] = field(default=False, metadata={"help": "Whether to use fully-connected attention"})
    no_causal_attention: Optional[bool] = field(default=False, metadata={"help": "Whether to remove the extra layer of causal attention"})
    no_gates: Optional[bool] = field(default=False, metadata={"help": "Whether to remove gate"})
    adapter_hidden_size: Optional[int] = field(default=768, metadata={"help": "Hidden size of adapter"})
    num_attention_heads: Optional[int] = field(default=12, metadata={"help": "Hidden size of adapter"})
    num_layers: Optional[int] = field(default=4, metadata={"help": "Number of layers for adapter"})
    use_last_layer: Optional[bool] = field(default=False, metadata={"help": "Hidden size of adapter"})
    test_subset: Optional[int] = field(default=None, metadata={"help": "Subset test set"})
    use_cached_results: Optional[bool] = field(default=False, metadata={"help": "Use cached results for evaluation."})
    
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

def init_trainer(script_args: ScriptArguments,
                 train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Dataset] = None,
                 compute_metrics: Optional[Callable] = None,
                 data_collator: Optional[Callable] = None,
                 eval_only: bool = False):
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
    
    # Number of layers to insert the adapters
    final_layer = 32 if script_args.use_last_layer else 31
    layers = list(range(max(0, final_layer - script_args.num_layers), final_layer))
    
    # Ensure adapter hidden size divides the number of attention heads
    assert script_args.adapter_hidden_size % script_args.num_attention_heads == 0, \
            "Adapter hidden size not divisible by number of attention heads!"

    # Initialize adapter configs
    if script_args.adapter_method == 'attention':
        use_gates = (not script_args.no_gates)
        adapter_config = AttentionAdapterConfig(adapter_type=script_args.adapter_type,
                                                injection_location=script_args.injection_location,
                                                use_gates=use_gates,
                                                gate_type=script_args.adapter_gate_type,
                                                hidden_size=script_args.adapter_hidden_size,
                                                layers=layers,
                                                num_attention_heads=script_args.num_attention_heads)
    elif script_args.adapter_method == 'structured':
        hierarchical_attention = (not script_args.full_attention)
        causal_attention = (not script_args.no_causal_attention)
        use_gates = (not script_args.no_gates)
        adapter_config = StructuredAdapterConfig(gate_type=script_args.adapter_gate_type,
                                                 pooling_method=script_args.pooling_method,
                                                 injection_location=script_args.injection_location,
                                                 adapter_type=script_args.adapter_type,
                                                 hierarchical_attention=hierarchical_attention,
                                                 causal_attention=causal_attention,
                                                 use_gates=use_gates,
                                                 hidden_size=script_args.adapter_hidden_size,
                                                 num_attention_heads=script_args.num_attention_heads,
                                                 layers=layers)
    elif script_args.adapter_method == 'mlp':
        adapter_config = MLPAdapterConfig(adapter_type=script_args.adapter_type,
                                          injection_location=script_args.injection_location,
                                          hidden_size=script_args.adapter_hidden_size,
                                          layers=layers)
    else:
        adapter_config = None
    
    model = MistralForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
        use_cache=False,
        adapter_config=adapter_config,
    )

    
    model.enable_input_require_grads()
    if script_args.use_lora:
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, modules_to_save=['adapter'])
        model = get_peft_model(model, peft_config)
    elif script_args.use_prefix:
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=100,
            token_dim=model.config.hidden_size,
            num_transformer_submodules=1,
            num_attention_heads=model.config.num_attention_heads,
            num_layers=model.config.num_hidden_layers,
            inference_mode=False
        )
        model = get_peft_model(model, peft_config)
    elif script_args.use_ia3:
        peft_config = IA3Config(
            peft_type="IA3",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj", "down_proj"],
            feedforward_modules=["down_proj"],
            inference_mode=False,
        )
        model = get_peft_model(model, peft_config)
    elif script_args.use_ptuning:
        peft_config = PromptEncoderConfig(
            peft_type="P_TUNING",
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=100,
            num_transformer_submodules=1,
            num_layers=2,
            encoder_reparameterization_type="MLP",
            encoder_hidden_size=768,
        )
        model = get_peft_model(model, peft_config)
    else:
        for name, module in model.named_children():
            for param in module.parameters():
                param.requires_grad = False

    if script_args.adapter_method is not None:
        # if script_args.adapter_method != 'structured':
        #     peft_module_casting_to_f16(model)
        for name, param in model.model.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True
    model.print_trainable_parameters()
    
    max_new_tokens = 300 if script_args.dataset_name == 'summscreen' else 100
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    prediction_loss_only = not eval_only
    # Step 3: Define the training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        per_device_eval_batch_size=script_args.eval_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        report_to=script_args.report_to,
        save_strategy=script_args.save_strategy,
        save_steps=script_args.save_steps,
        evaluation_strategy=script_args.evaluation_strategy,
        eval_steps=script_args.eval_steps,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        fp16=True,
        fp16_full_eval=True,
        logging_first_step=True,
        generation_config=generation_config,
        predict_with_generate=True,
        remove_unused_columns=True,
        prediction_loss_only=prediction_loss_only,
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

class StopSequenceRepeatCriteria(StoppingCriteria):

    def __init__(self, tokenizer, stop_sequences=[], max_repeat=None):
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences
        self.max_repeat = max_repeat

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # not compatible with batch size > 1
        if input_ids.shape[0] > 1:
            return False
        
        input_ids = input_ids[0]
        
        # Check if last `max_repeat` tokens are the same
        if torch.unique(input_ids[-self.max_repeat:]).numel() == 1:
            return True

        # Check if generation ends with any of the `stop_sequences ``
        decoded_string = self.tokenizer.decode(input_ids)
        for stop in self.stop_sequences:
            if decoded_string.endswith(stop):
                return True
        return False

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Step 1: Load the dataset
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2',  'rougeL', 'rougeLsum'], use_stemmer=False)
    def summarization_metrics(eval_preds):
        preds, labels = eval_preds

        # decode preds and labels
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # rougeLSum expects newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        aggregator = scoring.BootstrapAggregator()
        result = {k: [] for k in scorer.rouge_types}
        for label, pred in zip(decoded_labels, decoded_preds):
            score = scorer.score(label, pred)
            aggregator.add_scores(score)
        result = aggregator.aggregate()
        for key in result:
            result[key] = result[key].mid.fmeasure
        return result
    
    def qa_metrics(eval_preds):
        preds, labels = eval_preds
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        f1_scores = [f1_score(pred, label) for pred, label in zip(decoded_preds, decoded_labels)]
        exact_match_scores = [exact_match_score(pred, label) for pred, label in zip(decoded_preds, decoded_labels)]
        return {'f1': 100 * np.mean(f1_scores),
                'exact_match': 100 * np.mean(exact_match_scores)}
    
    
    return_token_type_ids = (script_args.adapter_method == 'structured')
    if script_args.dataset_name == 'samsum':
        dataset = get_preprocessed_samsum(tokenizer, return_token_type_ids=return_token_type_ids)
        compute_metrics = summarization_metrics
    elif script_args.dataset_name == 'summscreen':
        dataset = get_preprocessed_summscreen(tokenizer, return_token_type_ids=return_token_type_ids)
        compute_metrics = summarization_metrics
    elif script_args.dataset_name == 'mediasum':
        dataset = get_preprocessed_mediasum(tokenizer, return_token_type_ids=return_token_type_ids)
        compute_metrics = summarization_metrics
    elif script_args.dataset_name == 'friendsqa':
        dataset = get_preprocessed_friendsqa(tokenizer, return_token_type_ids=return_token_type_ids)
        compute_metrics = qa_metrics
    else:
        raise NotImplementedError(f"Dataset {script_args.dataset_name} not supported.")

    data_collator = DataCollatorForSeq2Seq(tokenizer)
    
    stopping_criteria = StopSequenceRepeatCriteria(tokenizer=tokenizer,
                                                   stop_sequences=['\n', '\r', '  '],
                                                   max_repeat=4)

    # Initialize the trainer
    if script_args.do_train:
        trainer = init_trainer(script_args,
                               dataset['train'],
                               dataset['validation'],
                               compute_metrics,
                               data_collator)
        trainer.train(resume_from_checkpoint=script_args.resume_from_checkpoint)
        trainer.accelerator.wait_for_everyone()
        
        # prediction_output = trainer.predict(test_dataset=dataset['test'],
        #                                     stopping_criteria=StoppingCriteriaList([stopping_criteria]))
        # if trainer.accelerator.is_main_process:
        #     print(prediction_output)
        #     torch.save(prediction_output, os.path.join(script_args.output_dir, 'evaluation_output.pt'))
        del trainer
    
    if script_args.do_eval:
        test_dataset = dataset['test']
        if script_args.test_subset is not None:
            test_dataset = Subset(test_dataset, list(range(script_args.test_subset)))
        
        
        if script_args.checkpoint_dir is None:
            checkpoint_paths = [os.path.join(script_args.output_dir, dir) for dir in os.listdir(script_args.output_dir)]
            checkpoint_paths = [path for path in checkpoint_paths if os.path.isdir(path)]
            checkpoint_paths = sorted(checkpoint_paths, key=lambda x: int(x.split('-')[-1]), reverse=True)
        else:
            checkpoint_paths = [os.path.join(script_args.output_dir, script_args.checkpoint_dir)]
            
        for checkpoint_path in checkpoint_paths:
            
            output_file = 'evaluation_output.pt' if script_args.test_subset is None else f"evaluation_output_{script_args.test_subset}.pt"
            prediction_output_file = os.path.join(checkpoint_path, output_file)
            if not os.path.isfile(prediction_output_file) or not script_args.use_cached_results:
                
                # Initialize a new trainer due to bug in loading checkpoint after `predict`
                torch.cuda.empty_cache()
                gc.collect()
                trainer = init_trainer(script_args,
                                       dataset['train'],
                                       dataset['validation'],
                                       compute_metrics,
                                       data_collator,
                                       eval_only=True)

                # TODO: This should be refactored into method `predict_from_checkpoint`
                trainer._load_from_checkpoint(checkpoint_path)
                print(f"Successfully loaded checkpoint from \"{checkpoint_path}\".")
                prediction_output = trainer.predict(test_dataset=test_dataset,
                                                    stopping_criteria=StoppingCriteriaList([stopping_criteria]))

                if trainer.accelerator.is_main_process:
                    print(f"Prediction results from checkpoint: \"{checkpoint_path}\"")
                    print(prediction_output)
                    torch.save(prediction_output, prediction_output_file)

                trainer.accelerator.wait_for_everyone()
                del trainer

            else:
                prediction_output = torch.load(prediction_output_file)
                print(f"Successfully loaded prediction output: \"{prediction_output_file}\"")
                
            # Save generated results in JSON format
            preds = prediction_output.predictions
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)
            results = []
            for idx, example in enumerate(test_dataset):
                if script_args.dataset_name == 'samsum':
                    result = {
                        'dialogue': example['dialogue'],
                        'summary': example['summary'],
                        'prediction': predictions[idx],
                    }
                elif script_args.dataset_name == 'summscreen':
                    result = {
                        'dialogue': '\n'.join(example['Transcript']),
                        'summary': ' '.join(example['Recap']),
                        'prediction': predictions[idx],
                    }
                elif script_args.dataset_name == 'mediasum':
                    result = {
                        'dialogue': example['document'],
                        'summary': example['summary'],
                        'prediction': predictions[idx].split('Summary: ')[-1],
                    }
                elif script_args.dataset_name == 'friendsqa':
                    prediction = predictions[idx]
                    answers = example['answers']
                    result = {
                        'dialogue': '\n'.join(example['dialogue']),
                        'question': example['question'],
                        'answers': answers,
                        'prediction': prediction,
                        'exact_match': metric_max_over_ground_truths(exact_match_score, prediction, answers),
                        'f1': metric_max_over_ground_truths(f1_score, prediction, answers),
                    }
                results.append(result)
            
            # Recompute results for all possible answers
            if script_args.dataset_name == 'friendsqa':
                avg_exact_match_score = 100 * (len([res for res in results if res['exact_match']]) / len(results))
                avg_f1_score = 100 * np.mean([res['f1'] for res in results])
                print(f"[Over All Answers] Exact Match: {avg_exact_match_score}, F1: {avg_f1_score}")

            with open(os.path.join(checkpoint_path, 'prediction_results.json'), 'w+') as f:
                json.dump(results, f, indent=4)
