from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from DeepPastChallenge.train import run_training  # noqa: E402

BASE_ARGS_PATH = os.getenv("BASE_ARGS_PATH")


def _load_overrides(overrides: dict[str, Any] | str | Path | None) -> dict[str, Any] | None:
    if overrides is None:
        return None
    if isinstance(overrides, dict):
        return overrides
    path = Path(overrides)
    if not path.is_file():
        raise FileNotFoundError(f"overrides yaml not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("overrides yaml must be a mapping")
    return data


def train_or_predict(*, overrides: dict[str, Any] | str | Path | None = None) -> dict[str, Any]:
    cfg = _load_overrides(overrides)
    return run_training(cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--overrides", type=str, default="")
    args = parser.parse_args()
    overrides_path = args.overrides.strip()
    run_args = _load_overrides(overrides_path) if overrides_path else None

    if BASE_ARGS_PATH:
        base = _load_overrides(BASE_ARGS_PATH) or {}
        if run_args:
            base.update(run_args)
        overrides = base
    else:
        overrides = run_args

    train_or_predict(overrides=overrides)
