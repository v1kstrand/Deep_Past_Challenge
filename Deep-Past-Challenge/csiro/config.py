from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_SEED: int = 42

_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_ROOT: str | None = os.getenv("DEFAULT_DATA_ROOT")
DEFAULT_OUTPUT_DIR: str = os.getenv(
    "DEFAULT_OUTPUT_DIR",
    str(_REPO_ROOT / "outputs"),
)

DEFAULTS: dict[str, Any] = dict(
    # Data
    data_root=DEFAULT_DATA_ROOT,
    train_path="train.csv",
    test_path="test.csv",
    id_col="id",
    src_col="transliteration",
    tgt_col="translation",
    val_split=0.1,
    seed=DEFAULT_SEED,
    # Model
    model_name="google/byt5-small",
    tokenizer_name=None,
    max_source_len=256,
    max_target_len=256,
    gen_max_len=256,
    num_beams=4,
    # Training
    device="cuda",
    epochs=3,
    batch_size=8,
    eval_batch_size=8,
    lr=5e-5,
    weight_decay=0.0,
    warmup_steps=0,
    grad_clip=1.0,
    trainable_dtype="fp16",
    # Logging / outputs
    log_every=50,
    eval_every=1,
    save_best_only=True,
    output_dir=DEFAULT_OUTPUT_DIR,
    run_name="baseline",
)


def default_num_workers(reserve: int = 2) -> int:
    n = (os.cpu_count() or 0) - int(reserve)
    return max(0, n)


def parse_dtype(dtype: str):
    import torch

    s = str(dtype).strip().lower()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype}")
