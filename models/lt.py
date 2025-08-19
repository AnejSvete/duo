import math
import typing

import einops
import flash_attn
import flash_attn.layers.rotary
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool,
) -> torch.Tensor:
    """
    A helper function that fuses bias addition, dropout, residual connection, and scaling.
    This is used in the transformer blocks.
    """
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)

    if residual is not None:
        out = residual + out
    return out


# function overload
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    """JIT-compiled version of bias_dropout_add_scale for training."""
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    """JIT-compiled version of bias_dropout_add_scale for inference."""
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
    """
    Implements Rotary Positional Embeddings (RoPE).
    RoPE is a modern technique for encoding positional information in transformers.
    """

    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.cos_cached[:, :, 2, :, :].fill_(1.0)
            self.sin_cached[:, :, 2, :, :].fill_(0.0)

        return self.cos_cached, self.sin_cached


def apply_rotary_pos_emb(qkv, cos, sin):
    cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]
    sin = sin[0, :, 0, 0, : sin.shape[-1] // 2]
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


def split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin):
    with torch.cuda.amp.autocast(enabled=False):
        cos, sin = rotary_cos_sin
        cos = cos.to(qkv.dtype)
        sin = sin.to(qkv.dtype)
        cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]
        sin = sin[0, :, 0, 0, : sin.shape[-1] // 2]
        q, k, v = qkv.chunk(3, dim=2)
        q = flash_attn.layers.rotary.apply_rotary_emb_torch(q.squeeze(dim=2), cos, sin)
        k = flash_attn.layers.rotary.apply_rotary_emb_torch(k.squeeze(dim=2), cos, sin)
        v = v.squeeze(dim=2)
    return q, k, v


def regular_attention_multi_headed(q, k, v):
    attention_output = F.scaled_dot_product_attention(
        query=q.transpose(1, 2),
        key=k.transpose(1, 2),
        value=v.transpose(1, 2),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )
    attention_output = attention_output.transpose(1, 2)
    return einops.rearrange(attention_output, "b s h d -> b s (h d)")


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        if x.ndim == 2:
            return self.embedding[x]
        assert x.ndim == 3
        return torch.einsum(
            "blv,ve->ble",
            torch.nn.functional.softmax(x, dim=-1).float(),
            self.embedding.float(),
        ).to(x.dtype)


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
