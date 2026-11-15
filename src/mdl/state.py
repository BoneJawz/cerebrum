"""Versioned, architecture-checked Brian2 continuation checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

def _manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def save_state(network, path: str | Path) -> tuple[Path, Path]:
    """Save Brian2's full continuation state and a readable compatibility manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    network.net.store("mdl_state", filename=str(path))
    manifest = _manifest_path(path)
    manifest.write_text(json.dumps(network.config.manifest(), indent=2) + "\n", encoding="utf-8")
    return path, manifest


def restore_state(network, path: str | Path) -> None:
    """Restore only when architecture and numerical configuration match exactly."""
    path = Path(path)
    manifest = _manifest_path(path)
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    expected = network.config.manifest()
    if saved != expected:
        raise ValueError("checkpoint configuration does not match this network")
    network.net.restore("mdl_state", filename=str(path), restore_random_state=True)
