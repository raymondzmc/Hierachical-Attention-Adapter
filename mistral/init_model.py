import torch
from peft import (
    get_peft_model,
    TaskType,
    LoraConfig,
    PromptTuningConfig,
    IA3Config,
    PromptEncoderConfig
)
from utils.arguments import ScriptArguments
from mistral.configuration_mistral import AttentionAdapterConfig, StructuredAdapterConfig, MLPAdapterConfig
from mistral.modeling_mistral import MistralForCausalLM

def init_mistral_model(script_args: ScriptArguments):
    
    # Number of layers to insert the adapters (TODO: remove hard code for Mistral-7B)
    final_layer = 32 if script_args.use_last_layer else 31
    if script_args.num_layers == 'all':
        num_layers = 32
    elif isinstance(script_args.num_layers, int):
        num_layers = script_args.num_layers
    else:
        raise ValueError(f"Unexpected value \"{script_args.num_layers}\" for --num_layers!")
    layers = list(range(max(0, final_layer - num_layers), final_layer))
    
    # Ensure adapter hidden size divides the number of attention heads
    if script_args.adapter_method in ['attention', 'structured']:
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
        use_gates = (not script_args.no_gates)
        adapter_config = StructuredAdapterConfig(gate_type=script_args.adapter_gate_type,
                                                 pooling_method=script_args.pooling_method,
                                                 injection_location=script_args.injection_location,
                                                 adapter_type=script_args.adapter_type,
                                                 hierarchical_attention=hierarchical_attention,
                                                 causal_attention=script_args.causal_attention,
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
        
    # TODO: Support quantized fine-tuning
    # if script_args.load_in_8bit and script_args.load_in_4bit:
    #     raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    # elif script_args.load_in_8bit or script_args.load_in_4bit:
    #     quantization_config = BitsAndBytesConfig(
    #         load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
    #     )
    #     # Copy the model to each device
    #     device_map = {"": Accelerator().local_process_index}
    #     torch_dtype = torch.float16
    # else:
    device_map = None
    quantization_config = None
    torch_dtype = torch.float16
    
    
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
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            modules_to_save=['adapter']
        )
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
    return model