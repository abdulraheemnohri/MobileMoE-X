"""A compact decoder-only sparse Mixture-of-Experts language model."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import MiniMoEConfig


class SparseMoE(nn.Module):
    """Top-k token routing to independent expert MLPs."""

    def __init__(self, config: MiniMoEConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.experts_per_token
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, config.d_ff),
                nn.GELU(),
                nn.Linear(config.d_ff, config.d_model),
            )
            for _ in range(config.n_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq, width = x.shape
        flat = x.reshape(-1, width)
        probabilities = self.router(flat).softmax(dim=-1)
        top_weights, top_indices = torch.topk(probabilities, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token_rows, slots = torch.where(top_indices == expert_id)
            if token_rows.numel() == 0:
                continue
            expert_input = flat.index_select(0, token_rows)
            expert_output = expert(expert_input)
            weights = top_weights[token_rows, slots].unsqueeze(-1)
            output.index_add_(0, token_rows, expert_output * weights)

        # Switch-style balancing term: encourages balanced probability and dispatch.
        mean_probability = probabilities.mean(dim=0)
        dispatch_fraction = F.one_hot(top_indices, self.n_experts).float().mean(dim=(0, 1))
        auxiliary_loss = self.n_experts * torch.sum(mean_probability * dispatch_fraction)
        return output.reshape(batch, seq, width), auxiliary_loss


class CausalSelfAttention(nn.Module):
    def __init__(self, config: MiniMoEConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            split_heads(q), split_heads(k), split_heads(v),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, seq, width)
        return self.proj(attended)


class MiniMoEBlock(nn.Module):
    def __init__(self, config: MiniMoEConfig):
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.norm_moe = nn.LayerNorm(config.d_model)
        self.moe = SparseMoE(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attention(self.norm_attention(x))
        moe_output, auxiliary_loss = self.moe(self.norm_moe(x))
        return x + moe_output, auxiliary_loss


class MiniMoEForCausalLM(nn.Module):
    def __init__(self, config: MiniMoEConfig | None = None):
        super().__init__()
        self.config = (config or MiniMoEConfig()).validate()
        c = self.config
        self.token_embedding = nn.Embedding(c.vocab_size, c.d_model)
        self.position_embedding = nn.Embedding(c.context_length, c.d_model)
        self.blocks = nn.ModuleList([MiniMoEBlock(c) for _ in range(c.n_layers)])
        self.final_norm = nn.LayerNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.context_length:
            raise ValueError("input sequence exceeds configured context_length")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        auxiliary_losses = []
        for block in self.blocks:
            x, aux = block(x)
            auxiliary_losses.append(aux)
        logits = self.lm_head(self.final_norm(x))
        aux_loss = torch.stack(auxiliary_losses).mean() if auxiliary_losses else logits.new_zeros(())
        return logits, aux_loss

    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, *, max_new_tokens: int = 80,
        temperature: float = 0.8, top_k: int = 40
    ) -> torch.Tensor:
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.context_length:]
            logits, _ = self(context)
            next_logits = logits[:, -1, :]
            if temperature == 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k > 0:
                    values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < values[:, [-1]]] = -torch.inf
                probabilities = next_logits.softmax(dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids
