"""Small, reproducible MiniMoE configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class MiniMoEConfig:
    vocab_size: int = 256
    context_length: int = 256
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_experts: int = 4
    experts_per_token: int = 2
    d_ff: int = 256
    dropout: float = 0.0
    router_aux_loss_coef: float = 0.01

    def validate(self) -> "MiniMoEConfig":
        if self.vocab_size != 256:
            raise ValueError("MiniMoE v1 uses the fixed 256-token UTF-8 byte vocabulary")
        if self.context_length < 2 or self.d_model < 8:
            raise ValueError("context_length must be >=2 and d_model must be >=8")
        if self.n_layers < 1 or self.n_heads < 1 or self.d_model % self.n_heads:
            raise ValueError("invalid layer/head configuration")
        if self.n_experts < 2 or not 1 <= self.experts_per_token <= self.n_experts:
            raise ValueError("experts_per_token must be between 1 and n_experts")
        if self.d_ff < 1 or not 0 <= self.dropout < 1:
            raise ValueError("invalid feed-forward size or dropout")
        return self

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MiniMoEConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data).validate()
