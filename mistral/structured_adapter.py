import torch
import torch.nn as nn
from typing import Optional, Tuple

from .configuration_mistral import MistralAdapterConfig
from transformers import BartModel
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
import pdb

class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.adapter_k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.adapter_v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.adapter_q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.adapter_out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()
        # get query proj
        query_states = self.adapter_q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.adapter_k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.adapter_v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.adapter_k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.adapter_v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.adapter_k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.adapter_v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.adapter_out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class MistralStructuredAdapter(nn.Module):
    def __init__(self, model_hidden_size: int, config: MistralAdapterConfig, layer_idx=None):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layer_idx = layer_idx
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
        
        if config.use_gate:
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

            for uid in unique_ids:
                mask = token_type_ids[i] == uid
                try:
                    # self.fc1(hidden_states)
                    # TODO: This causes inplace operation (no grad)
                    selected_states = hidden_states[i, mask]
                except:
                    pdb.set_trace()
                if self.config.turn_embedding_method == 'mean':
                    turn_embedding = selected_states.mean(dim=0)
                else:
                    raise NotImplementedError(
                        f"Unsupported method: {self.config.turn_embedding_method}.")
                dialogue_hidden_states.append(turn_embedding)
            dialogue_hidden_states = torch.stack(dialogue_hidden_states).unsqueeze(0)
            
            summary_mask = (token_type_ids[i] == -1)
            
            summary_hidden_states = hidden_states[i, summary_mask].unsqueeze(0)
            summary_len = summary_hidden_states.shape[1]
            
            # Dialogue self-attention
            dialogue_hidden_states = self.fc1(dialogue_hidden_states)
            dialogue_hidden_states = self.dialogue_self_attention(dialogue_hidden_states)[0]
            dialogue_hidden_states = nn.functional.dropout(dialogue_hidden_states, p=self.dropout, training=self.training)
            dialogue_hidden_states = self.dialogue_self_attention_layer_norm(dialogue_hidden_states)
            
            
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
                    gate = torch.sigmoid(self.output_gate(hidden_states[i, summary_mask]))
                elif self.gate_type == 'tanh':
                    gate = torch.tanh(self.output_gate(hidden_states[i, summary_mask]))
                summary_hidden_states = gate * summary_hidden_states

            output.append((summary_hidden_states.to(hidden_states.dtype), summary_mask))
        
        assert len(output) == batch_size
        return output