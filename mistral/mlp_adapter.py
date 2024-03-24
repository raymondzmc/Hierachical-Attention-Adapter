import torch
import torch.nn as nn
import pdb


class MLPAdapter(nn.Module):
    def __init__(self, model_hidden_size, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.fc1 = nn.Linear(model_hidden_size, config.hidden_size)
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear(config.hidden_size, model_hidden_size)
        
        self.gradient_checkpointing = True

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states