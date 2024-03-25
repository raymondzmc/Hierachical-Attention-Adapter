import torch
import torch.nn as nn
from .attention import Attention
import pdb


class AttentionAdapter(nn.Module):
    def __init__(self, model_hidden_size, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layer_idx = layer_idx

        self.fc1 = nn.Linear(model_hidden_size, config.hidden_size)
        self.attn = Attention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
        )
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size)

        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.SiLU()
        self.fc3 = nn.Linear(config.hidden_size, model_hidden_size)
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

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.attn(hidden_states)[0]
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = self.attention_layer_norm(hidden_states)
        
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc3(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.final_layer_norm(hidden_states)
        
        if self.output_gate is not None:
            if self.gate_type == 'sigmoid':
                gate = torch.sigmoid(self.output_gate(hidden_states))
            elif self.gate_type == 'tanh':
                gate = torch.tanh(self.output_gate(hidden_states))
            hidden_states = gate * hidden_states
        return hidden_states