import torch
import torch.nn as nn
from typing import Optional, Tuple

from .configuration_mistral import StructuredAdapterConfig
from .attention import Attention, DialogueTurnAttention
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
import pdb


class MistralStructuredAdapter(nn.Module):
    def __init__(self, model_hidden_size: int, config: StructuredAdapterConfig, layer_idx=None):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layer_idx = layer_idx
        self.pooling_method = config.pooling_method
        # self.causal_self_attention = config.causal_self_attention
        
        if self.pooling_method == "attention":
            self.turn_self_attention = DialogueTurnAttention(model_hidden_size, config.hidden_size)
        
        self.fc1 = nn.Linear(model_hidden_size, config.hidden_size)
        self.dialogue_self_attention = Attention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dialogue_self_attention_layer_norm = nn.LayerNorm(config.hidden_size)
        
        self.fc2 = nn.Linear(model_hidden_size, config.hidden_size)
        self.summary_self_attention = Attention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
        )
        self.summary_self_attention_layer_norm = nn.LayerNorm(config.hidden_size)
        
        self.cross_attention = Attention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.cross_attention_layer_norm = nn.LayerNorm(config.hidden_size)
        self.fc3 = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.SiLU()
        self.fc4 = nn.Linear(config.hidden_size, model_hidden_size)
        self.final_layer_norm = nn.LayerNorm(model_hidden_size)
        
        if config.use_gates:
            # self.output_gate = nn.Parameter(torch.zeros((model_hidden_size)), requires_grad=True)
            self.gate_type = config.gate_type
            self.output_gate = nn.Sequential(nn.Linear(model_hidden_size, 256),
                                             nn.SiLU(),
                                             nn.Linear(256, 1))
        else:
            self.output_gate = None


        self.gradient_checkpointing = True
    
    def forward(self, hidden_states: torch.Tensor, token_type_ids: torch.LongTensor):
        batch_size = hidden_states.shape[0]

        # Create turn embeddings by aggregating hidden states by their token_type_ids
        output = []
        for i in range(batch_size):
            unique_ids = torch.unique(token_type_ids[i][token_type_ids[i] > 0])
            dialogue_hidden_states = []

            if self.config.hierarchical_attention:
                turn_masks = []
                for uid in unique_ids:
                    mask = token_type_ids[i] == uid
                    turn_masks.append(mask)

                turn_masks = torch.stack(turn_masks)

                if self.pooling_method == 'attention':
                    turn_masks = turn_masks.transpose(0, 1)
                    dialogue_hidden_states = self.turn_self_attention(hidden_states[i].unsqueeze(0), turn_masks.unsqueeze(0))
                elif self.pooling_method == 'mean':
                    dialogue_hidden_states = torch.stack([hidden_states[i, mask].mean(dim=0) for mask in turn_masks]).unsqueeze(0)
                elif self.pooling_method == 'last':
                    dialogue_hidden_states = torch.stack([hidden_states[i, mask][-1] for mask in turn_masks]).unsqueeze(0)
            else:
                dialogue_hidden_states = hidden_states[i, token_type_ids[i] != -1].unsqueeze(0)

            # Dialogue self-attention
            dialogue_hidden_states = self.fc1(dialogue_hidden_states)
            dialogue_hidden_states = self.dialogue_self_attention(dialogue_hidden_states)[0]
            dialogue_hidden_states = nn.functional.dropout(dialogue_hidden_states, p=self.dropout, training=self.training)
            dialogue_hidden_states = self.dialogue_self_attention_layer_norm(dialogue_hidden_states)
            

            summary_mask = (token_type_ids[i] == -1)
            summary_hidden_states = hidden_states[i].unsqueeze(0)
            summary_len = summary_hidden_states.shape[1]

            # Summary self-attention
            causal_attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask=None,
                input_shape=(1, summary_len),
                inputs_embeds=summary_hidden_states,
                past_key_values_length=0,
                sliding_window=4096)

            summary_hidden_states = self.fc2(summary_hidden_states)
            summary_hidden_states = self.summary_self_attention(summary_hidden_states, attention_mask=causal_attention_mask)[0]
            summary_hidden_states = nn.functional.dropout(summary_hidden_states, p=self.dropout, training=self.training)
            summary_hidden_states = self.summary_self_attention_layer_norm(summary_hidden_states)
            
            # Summary cross-attention
            summary_hidden_states = self.cross_attention(summary_hidden_states, dialogue_hidden_states)[0]
            summary_hidden_states = nn.functional.dropout(summary_hidden_states, p=self.dropout, training=self.training)
            summary_hidden_states = self.cross_attention_layer_norm(summary_hidden_states)
            
            # MLP
            summary_hidden_states = self.fc3(summary_hidden_states)
            summary_hidden_states = self.activation(summary_hidden_states)
            summary_hidden_states = self.fc4(summary_hidden_states)
            summary_hidden_states = nn.functional.dropout(summary_hidden_states, p=self.dropout, training=self.training)
            summary_hidden_states = self.final_layer_norm(summary_hidden_states).squeeze(0)
            
            if self.output_gate is not None:
                if self.gate_type == 'sigmoid':
                    gate = torch.sigmoid(self.output_gate(hidden_states[i]))
                elif self.gate_type == 'tanh':
                    gate = torch.tanh(self.output_gate(hidden_states[i]))
                summary_hidden_states = gate * summary_hidden_states

            output.append((summary_hidden_states.to(hidden_states.dtype), summary_mask))
        
        return output