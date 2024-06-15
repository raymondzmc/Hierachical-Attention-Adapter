from typing import Optional
from dataclasses import dataclass, field, asdict, fields

SUPPORTED_DATASETS = ['samsum', 'summscreen', 'mediasum', 'friendsqa']
SUPPORTED_ADAPTER_METHODS = ['attention', 'structured', 'mlp']

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """
    config_file: Optional[str] = field(default=None, metadata={"help": "Path to configuration json file for storing the arguments"})
    seed: Optional[int] = field(default=42)
    data_seed: Optional[int] = field(default=42)
    model_name: Optional[str] = field(default="mistralai/Mistral-7B-v0.1", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default='samsum', metadata={"help": "the dataset name", "choices": SUPPORTED_DATASETS}
    )
    do_train: Optional[bool] = field(default=False, metadata={"help": "Whether to perform training"})
    do_eval: Optional[bool] = field(default=False, metadata={"help": "Whether to perform evaluation"})
    
    report_to: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    optim: Optional[str] = field(default="adamw_torch", metadata={"help": "The optimizer to use"})
    group_by_length: Optional[bool] = field(default=True, metadata={"help": "Whether or not to group together samples of roughly the same length in the training dataset"})
    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default='polynomial')
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
    adapter_method: Optional[str] = field(default=None, metadata={"help": "Adapter method to use", "choices": SUPPORTED_ADAPTER_METHODS})
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
    save_only_model: Optional[bool] = field(default=True, metadata={"help": "whether to only save the model"})

    # LoRA hyperparameters
    lora_r: Optional[int] = field(default=8)
    lora_alpha: Optional[int] = field(default=8)
    lora_dropout: Optional[int] = field(default=0.1)
    
    # Our defined arguments
    adapter_gate_type: Optional[str] = field(default='tanh', metadata={"choices": ['sigmoid', 'tanh']})
    pooling_method: Optional[str] = field(default='mean', metadata={"choices": ['mean', 'attention', 'last']})
    injection_location: Optional[str] = field(default='mlp', metadata={"choices": ['sa', 'mlp', 'both']})
    adapter_type: Optional[str] = field(default='parallel', metadata={"choices": ['parallel', 'sequential']})
    full_attention: Optional[bool] = field(default=False, metadata={"help": "Whether to use fully-connected attention"})
    causal_attention: Optional[bool] = field(default=True, metadata={"help": "Whether to add an extra layer of causal attention"})
    no_gates: Optional[bool] = field(default=False, metadata={"help": "Whether to remove gate"})
    adapter_hidden_size: Optional[int] = field(default=768, metadata={"help": "Hidden size of adapter"})
    num_attention_heads: Optional[int] = field(default=12, metadata={"help": "Number of attention heads of adapter"})
    num_layers: Optional[int] = field(default=32, metadata={"help": "Number of layers for adapter"})
    use_last_layer: Optional[bool] = field(default=False, metadata={"help": "Whether or not to add to the last layer"})
    test_subset: Optional[int] = field(default=None, metadata={"help": "Subset test set"})
    use_cached_results: Optional[bool] = field(default=False, metadata={"help": "Use cached results for evaluation."})
    
    def to_dict(self):
        return asdict(self)
    
    def update(self, other: 'ScriptArguments'):
        for field in fields(self):
            setattr(self, field.name, getattr(other, field.name))