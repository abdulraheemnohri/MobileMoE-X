"""Minimal local training loop for MiniMoE-X."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import MiniMoEConfig
from .data import ByteTextDataset
from .model import MiniMoEForCausalLM


def train(
    text_path: str | Path, output: str | Path, *, steps: int = 500,
    batch_size: int = 16, learning_rate: float = 3e-4, device: str = "cpu",
    config: MiniMoEConfig | None = None,
) -> dict:
    if steps < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("steps, batch_size and learning_rate must be positive")
    cfg = (config or MiniMoEConfig()).validate()
    target_device = torch.device(device)
    dataset = ByteTextDataset(text_path, cfg.context_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model = MiniMoEForCausalLM(cfg).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    step = 0
    running_loss = 0.0
    while step < steps:
        for x, y in loader:
            x, y = x.to(target_device), y.to(target_device)
            logits, aux_loss = model(x)
            language_loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
            loss = language_loss + cfg.router_aux_loss_coef * aux_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            running_loss += float(loss.detach())
            if step >= steps:
                break
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "mobilemoe-mini-v1",
        "config": cfg.__dict__,
        "model_state_dict": model.cpu().state_dict(),
        "training": {
            "steps": steps, "batch_size": batch_size,
            "learning_rate": learning_rate, "mean_loss": running_loss / steps,
            "device": str(target_device),
        },
    }, destination)
    return {"checkpoint": str(destination), "steps": steps, "mean_loss": running_loss / steps,
            "parameter_count": sum(p.numel() for p in model.parameters())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small MiniMoE model on a local text file")
    parser.add_argument("text", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/minimoe.pt"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    cfg = MiniMoEConfig.load(args.config) if args.config else MiniMoEConfig()
    print(json.dumps(train(args.text, args.output, steps=args.steps, batch_size=args.batch_size,
                           learning_rate=args.learning_rate, device=args.device, config=cfg), indent=2))


if __name__ == "__main__":
    main()
