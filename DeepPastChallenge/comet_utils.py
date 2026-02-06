from __future__ import annotations

import json
import os
from typing import Any


def _enabled(cfg: dict[str, Any]) -> bool:
    val = cfg.get("comet_project_name", None)
    if val is None:
        return False
    if isinstance(val, str) and not val.strip():
        return False
    return True


def maybe_init_comet(cfg: dict[str, Any]):
    """
    Initialize Comet logging if enabled.

    Enable rule:
    - on if cfg['comet_project_name'] is not None/empty

    Auth rule:
    - COMET_API_KEY must be present in env (hard fail if enabled and missing)
    """
    if not _enabled(cfg):
        return None

    api_key = os.getenv("COMET_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Comet enabled but COMET_API_KEY is not set in the environment.")

    try:
        from comet_ml import Experiment, ExistingExperiment  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Comet enabled but comet-ml is not installed. Install with: pip install -e '.[comet]'"
        ) from exc

    project_name = str(cfg["comet_project_name"]).strip()
    experiment_key = cfg.get("comet_experiment_key", None)
    experiment_name = str(cfg.get("run_name") or "").strip() or "run"

    if experiment_key:
        exp = ExistingExperiment(
            api_key=api_key,
            project_name=project_name,
            previous_experiment=str(experiment_key),
            auto_output_logging="simple",
        )
    else:
        exp = Experiment(
            api_key=api_key,
            project_name=project_name,
            auto_output_logging="simple",
        )

    exp.set_name(experiment_name)

    # Log full config as params and as an asset for reproducibility.
    try:
        exp.log_parameters(cfg)
    except Exception:
        exp.log_parameter("config_json", json.dumps(cfg, sort_keys=True, default=str))

    try:
        exp.log_asset_data(
            json.dumps(cfg, indent=2, sort_keys=True, default=str),
            name="config.json",
        )
    except Exception:
        pass

    return exp
