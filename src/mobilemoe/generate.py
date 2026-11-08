"""Load a MiniMoE checkpoint and generate text."""
from __future__ import annotations

import argparse
import torch

from .config import MiniMoEConfig
from .model import MiniMoEForCausalLM
from .tokenizer import ByteTokenizer


def generate_text(checkpoint: str, prompt: str, *, max_new_tokens: int = 100,
                  device: str = "cpu", temperature: float = 0.8) -> str:
    # weights_only=True avoids arbitrary pickle object loading on supported PyTorch versions.
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload.get("format") != "mobilemoe-mini-v1":
        raise ValueError("unsupported checkpoint format")
    config = MiniMoEConfig(**payload["config"]).validate()
    model = MiniMoEForCausalLM(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    tokenizer = ByteTokenizer()
    input_ids = torch.tensor([tokenizer.encode(prompt) or [32]], dtype=torch.long, device=device)
    output = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature)
    return tokenizer.decode(output[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with MiniMoE-X")
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(generate_text(args.checkpoint, args.prompt, max_new_tokens=args.tokens,
                        temperature=args.temperature, device=args.device))


if __name__ == "__main__":
    main()
