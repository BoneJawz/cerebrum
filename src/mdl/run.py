"""Minimal headless entry point for one event or a continuous event sequence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import ModelConfig
from .network import DistributedVisualAssociator
from .state import restore_state, save_state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cue", type=int, default=0, help="auditory spike channel")
    parser.add_argument("--visual-cells", default="", help="comma-separated active visual cell indices")
    parser.add_argument("--duration", type=float, default=0.060, help="biological seconds")
    parser.add_argument("--no-plasticity", action="store_true")
    parser.add_argument("--state-in", type=Path)
    parser.add_argument("--state-out", type=Path)
    args = parser.parse_args(argv)

    config = ModelConfig(seed=args.seed)
    model = DistributedVisualAssociator(config, learning=not args.no_plasticity)
    if args.state_in:
        restore_state(model, args.state_in)
    visual = None
    if args.visual_cells.strip():
        visual = np.zeros(config.visual_shape, dtype=np.uint8)
        for value in args.visual_cells.split(","):
            idx = int(value)
            if not 0 <= idx < config.visual_inputs:
                parser.error(f"visual cell {idx} outside [0, {config.visual_inputs})")
            visual.flat[idx] = 1
    response = model.run_event(cue=args.cue, visual=visual, duration_s=args.duration)
    if args.state_out:
        save_state(model, args.state_out)
    print(json.dumps({
        "architecture": config.architecture_version,
        "seed": config.seed,
        "simulation_time_s": float(model.net.t),
        "auditory_events": response["auditory_ids"].tolist(),
        "visual_events": response["visual_ids"].tolist(),
        "spike_counts": response["spike_counts"].tolist(),
        "peak_voltage_mv": response["peak_depolarization_mv"].tolist(),
        "weight_min": float(model.weights().min()),
        "weight_max": float(model.weights().max()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
