import typing

import einops
import flash_attn
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn

from .common import (
    bias_dropout_add_scale_fused_train,
    bias_dropout_add_scale_fused_inference,
    Rotary,
    apply_rotary_pos_emb,
    split_and_apply_rotary_pos_emb,
    regular_attention_multi_headed,
    LayerNorm,
    EmbeddingLayer,
)


#################################################################################
#                                 Core Model                                    #
#################################################################################


class LTBlockCausal(nn.Module):
    """Causal Transformer Block using Flash Attention."""

    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout = dropout

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin):
        batch_size, seq_len = x.shape[0], x.shape[1]
        bias_dropout_scale_fn = self._get_bias_dropout_scale()
        x_skip = x

        # Attention operation
        x = self.norm1(x)
        qkv = self.attn_qkv(x)
        qkv = einops.rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads
        )
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        qkv = einops.rearrange(qkv, "b s ... -> (b s) ...")
        cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * seq_len,
            step=seq_len,
            dtype=torch.int32,
            device=qkv.device,
        )
        x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0.0, causal=True
        )
        x = einops.rearrange(x, "(b s) h d -> b s (h d)", b=batch_size)
        scale = torch.ones(1, device=x.device, dtype=x.dtype)
        x = bias_dropout_scale_fn(self.attn_out(x), None, scale, x_skip, self.dropout)

        # MLP operation
        x = bias_dropout_scale_fn(self.mlp(self.norm2(x)), None, scale, x, self.dropout)
        return x


class LTBlock(nn.Module):
    """Non-Causal Transformer Block."""

    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout = dropout

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin):
        bias_dropout_scale_fn = self._get_bias_dropout_scale()
        x_skip = x

        # Attention operation
        x = self.norm1(x)
        qkv = einops.rearrange(
            self.attn_qkv(x),
            "b s (three h d) -> b s three h d",
            three=3,
            h=self.n_heads,
        )
        q, k, v = split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin)
        x = regular_attention_multi_headed(q, k, v)
        scale = torch.ones(1, device=x.device, dtype=x.dtype)
        x = bias_dropout_scale_fn(self.attn_out(x), None, scale, x_skip, self.dropout)

        # MLP operation
        x = bias_dropout_scale_fn(self.mlp(self.norm2(x)), None, scale, x, self.dropout)
        return x


class LTFinalLayer(nn.Module):
    """Final processing layer for the Looped Transformer."""

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class LT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    """
    Looped Transformer (LT) model.
    This model implements a transformer where a block of layers can be repeated
    dynamically based on the input sequence length.
    """

    def __init__(
        self,
        config,
        vocab_size: int,
        loop_depth_function: typing.Optional[typing.Callable[[int], int]] = None,
    ):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)

        if loop_depth_function is None:
            self.loop_depth_function = lambda n: 1
        else:
            self.loop_depth_function = loop_depth_function

        self.config = config
        self.vocab_size = vocab_size
        dim = config.model.hidden_size
        self.causal = config.algo.causal_attention

        # Initial layers (Block 'A')
        self.vocab_embed = EmbeddingLayer(dim, vocab_size)
        self.rotary_emb = Rotary(dim // config.model.n_heads)

        # Repeating layers (Block 'B')
        blocks = []
        for _ in range(config.model.n_blocks):
            if self.causal:
                block = LTBlockCausal(
                    dim=dim, n_heads=config.model.n_heads, dropout=config.model.dropout
                )
            else:
                block = LTBlock(
                    dim=dim,
                    n_heads=config.model.n_heads,
                    dropout=config.model.dropout,
                )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        # Final layer (Block 'C')
        self.output_layer = LTFinalLayer(
            hidden_size=dim,
            out_channels=vocab_size,
        )

    def forward(self, x, sigma=None):
        # 1. Initial Layers (Block A) - Executed once
        x = self.vocab_embed(x)
        rotary_cos_sin = self.rotary_emb(x)

        # Determine the number of loops based on sequence length
        seq_len = x.shape[1]
        num_loops = self.loop_depth_function(seq_len)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # 2. Repeated Layers (Block B) - Executed `num_loops` times
            for _ in range(num_loops):
                for block in self.blocks:
                    x = block(x, rotary_cos_sin)

            # 3. Final Layer (Block C) - Executed once
            x = self.output_layer(x)

        return x
