"""Command-line interface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .checkpoint import inspect_checkpoint
from .io import load_artifact, save_artifact
from .quantization import quantize_tensor, storage_report
from .profile import load_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mobilemoe")
    commands = parser.add_subparsers(dest="command", required=True)
    quant = commands.add_parser("quantize", help="quantize a .npy tensor")
    quant.add_argument("input", type=Path)
    quant.add_argument("output", type=Path)
    quant.add_argument("--bits", type=int, choices=(1, 2, 4), default=4)
    quant.add_argument("--group-size", type=int, default=128)
    inspect = commands.add_parser("inspect", help="inspect an artifact directory")
    inspect.add_argument("artifact", type=Path)
    checkpoint = commands.add_parser("inspect-checkpoint", help="inspect local safetensors metadata without loading weights")
    checkpoint.add_argument("directory", type=Path)
    checkpoint.add_argument("--output", type=Path, help="write inventory JSON to this path")
    profile = commands.add_parser("show-profile", help="show the bundled LFM2-700M profile")
    profile.add_argument("--file", type=Path, default=Path(__file__).resolve().parents[2] / "profiles" / "lfm2_700m.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "quantize":
        tensor = np.load(args.input, allow_pickle=False)
        artifact = quantize_tensor(tensor, bits=args.bits, group_size=args.group_size)
        save_artifact(artifact, args.output)
        print(json.dumps(storage_report(artifact), indent=2))
    elif args.command == "inspect":
        artifact = load_artifact(args.artifact)
        print(json.dumps(artifact.manifest() | storage_report(artifact), indent=2))
    elif args.command == "inspect-checkpoint":
        inventory = inspect_checkpoint(args.directory)
        rendered = json.dumps(inventory, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\\n", encoding="utf-8")
        print(rendered)
    elif args.command == "show-profile":
        print(json.dumps(load_profile(args.file), indent=2))


if __name__ == "__main__":
    main()
