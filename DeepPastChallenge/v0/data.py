from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SplitData:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame | None = None


def _resolve_path(root: str | None, path: str) -> str:
    if root is None:
        return path
    if path.startswith(("~", "/", "\\")):
        return path
    return str(Path(root) / path)


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def train_valid_split(
    df: pd.DataFrame,
    *,
    val_split: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1.")
    n = len(df)
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(n)
    n_val = int(round(n * val_split))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


class TextTranslationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        tokenizer,
        src_col: str,
        tgt_col: str | None,
        max_source_len: int,
        max_target_len: int,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.src_col = src_col
        self.tgt_col = tgt_col
        self.max_source_len = int(max_source_len)
        self.max_target_len = int(max_target_len)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[int(idx)]
        src = str(row[self.src_col])
        model_inputs = self.tokenizer(
            src,
            max_length=self.max_source_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in model_inputs.items()}
        if self.tgt_col is not None:
            tgt = str(row[self.tgt_col])
            try:
                labels = self.tokenizer(
                    text_target=tgt,
                    max_length=self.max_target_len,
                    truncation=True,
                    padding=False,
                    return_tensors="pt",
                )
            except TypeError:
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        tgt,
                        max_length=self.max_target_len,
                        truncation=True,
                        padding=False,
                        return_tensors="pt",
                    )
            item["labels"] = labels["input_ids"].squeeze(0).tolist()
        for k, v in list(item.items()):
            if isinstance(v, np.ndarray):
                item[k] = torch.from_numpy(v)
        return item


def build_splits(
    *,
    train_path: str,
    test_path: str | None,
    data_root: str | None,
    val_split: float,
    seed: int,
) -> SplitData:
    train_csv = _resolve_path(data_root, train_path)
    df = load_csv(train_csv)
    tr_df, va_df = train_valid_split(df, val_split=val_split, seed=seed)
    test_df = None
    if test_path:
        test_csv = _resolve_path(data_root, test_path)
        test_df = load_csv(test_csv)
    return SplitData(train=tr_df, valid=va_df, test=test_df)
