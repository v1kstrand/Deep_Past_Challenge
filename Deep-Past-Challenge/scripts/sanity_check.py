from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def _path(root: str | None, name: str) -> Path:
    if root is None:
        return Path(name)
    return Path(root) / name


def _print_basic(df: pd.DataFrame, name: str) -> None:
    print(f"== {name} ==")
    print("rows:", len(df))
    print("cols:", df.columns.tolist())
    print("nulls:", df.isna().sum().to_dict())
    print(df.head(2))
    print()


def _print_lengths(df: pd.DataFrame, col: str, label: str) -> None:
    lens = df[col].astype(str).str.len()
    pct = [50, 75, 90, 95, 99]
    stats = {p: int(lens.quantile(p / 100.0)) for p in pct}
    print(f"{label} length stats:", stats)


def run(data_root: str, train_name: str = "train.csv", test_name: str = "test.csv") -> None:
    train_path = _path(data_root, train_name)
    test_path = _path(data_root, test_name)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    _print_basic(train, "train")
    _print_basic(test, "test")

    if "transliteration" in train.columns:
        _print_lengths(train, "transliteration", "train.transliteration")
    if "translation" in train.columns:
        _print_lengths(train, "translation", "train.translation")
    if "transliteration" in test.columns:
        _print_lengths(test, "transliteration", "test.transliteration")


if __name__ == "__main__":
    data_root = os.getenv(
        "DATA_ROOT",
        "/root/.cache/kagglehub/competitions/deep-past-initiative-machine-translation",
    )
    run(data_root)
