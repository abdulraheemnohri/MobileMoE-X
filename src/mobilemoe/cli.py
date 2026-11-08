"""Command-line interface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .io import load_artifact, save_artifact
from .quantization import quantize_tensor, storage_report


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


if __name__ == "__main__":
    main()
