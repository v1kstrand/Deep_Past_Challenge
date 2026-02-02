from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
)

from .amp import autocast_context, grad_scaler
from .config import DEFAULTS, default_num_workers
from .data import TextTranslationDataset, build_splits
from .metrics import kaggle_mt_score


@dataclass
class TrainState:
    best_score: float = -1.0
    best_epoch: int = -1


def _resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def build_model_and_tokenizer(cfg: dict[str, Any]):
    model_name = str(cfg["model_name"])
    tokenizer_name = str(cfg.get("tokenizer_name") or model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer


def build_dataloaders(cfg: dict[str, Any], tokenizer, model):
    splits = build_splits(
        train_path=str(cfg["train_path"]),
        test_path=str(cfg.get("test_path") or ""),
        data_root=cfg.get("data_root"),
        val_split=float(cfg["val_split"]),
        seed=int(cfg["seed"]),
    )

    train_ds = TextTranslationDataset(
        splits.train,
        tokenizer=tokenizer,
        src_col=str(cfg["src_col"]),
        tgt_col=str(cfg["tgt_col"]),
        max_source_len=int(cfg["max_source_len"]),
        max_target_len=int(cfg["max_target_len"]),
    )
    valid_ds = TextTranslationDataset(
        splits.valid,
        tokenizer=tokenizer,
        src_col=str(cfg["src_col"]),
        tgt_col=str(cfg["tgt_col"]),
        max_source_len=int(cfg["max_source_len"]),
        max_target_len=int(cfg["max_target_len"]),
    )
    test_ds = None
    if splits.test is not None:
        test_ds = TextTranslationDataset(
            splits.test,
            tokenizer=tokenizer,
            src_col=str(cfg["src_col"]),
            tgt_col=None,
            max_source_len=int(cfg["max_source_len"]),
            max_target_len=int(cfg["max_target_len"]),
        )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    num_workers = int(cfg.get("num_workers") or default_num_workers())
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=int(cfg["eval_batch_size"]),
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
        )

    return splits, train_loader, valid_loader, test_loader


def _decode_labels(tokenizer, labels: torch.Tensor) -> list[str]:
    labels = labels.clone()
    labels[labels == -100] = tokenizer.pad_token_id
    return tokenizer.batch_decode(labels, skip_special_tokens=True)


def evaluate(
    model,
    tokenizer,
    data_loader: DataLoader,
    device: str,
    *,
    gen_max_len: int,
    num_beams: int,
) -> float:
    model.eval()
    preds: list[str] = []
    refs: list[str] = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="valid", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=int(gen_max_len),
                num_beams=int(num_beams),
            )
            preds.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
            refs.extend(_decode_labels(tokenizer, batch["labels"]))
    return kaggle_mt_score(preds, refs)


def train_one_epoch(
    model,
    data_loader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    device: str,
    *,
    grad_clip: float | None,
    log_every: int,
) -> float:
    model.train()
    total_loss = 0.0
    step = 0
    for batch in tqdm(data_loader, desc="train", leave=False):
        step += 1
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast_context(device):
            outputs = model(**batch)
            loss = outputs.loss
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        total_loss += float(loss.item())
        if log_every and step % int(log_every) == 0:
            avg = total_loss / max(step, 1)
            tqdm.write(f"step {step} loss {avg:.4f}")
    return total_loss / max(step, 1)


def save_checkpoint(model, tokenizer, output_dir: str, run_name: str, *, tag: str):
    path = os.path.join(output_dir, run_name, tag)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def predict_to_csv(
    model,
    tokenizer,
    data_loader: DataLoader,
    ids: Sequence[Any],
    *,
    device: str,
    gen_max_len: int,
    num_beams: int,
    output_path: str,
):
    model.eval()
    preds: list[str] = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="predict", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=int(gen_max_len),
                num_beams=int(num_beams),
            )
            preds.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    out = pd.DataFrame({"id": list(ids), "translation": preds})
    out.to_csv(output_path, index=False)


def run_training(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg: dict[str, Any] = dict(DEFAULTS)
    if overrides:
        cfg.update(overrides)

    device = _resolve_device(str(cfg.get("device", "cuda")))
    model, tokenizer = build_model_and_tokenizer(cfg)
    model.to(device)

    splits, train_loader, valid_loader, test_loader = build_dataloaders(cfg, tokenizer, model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    total_steps = int(cfg["epochs"]) * max(len(train_loader), 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg["warmup_steps"]),
        num_training_steps=total_steps,
    )
    scaler = grad_scaler(device)

    os.makedirs(str(cfg["output_dir"]), exist_ok=True)
    state = TrainState()
    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            grad_clip=cfg.get("grad_clip"),
            log_every=int(cfg.get("log_every") or 0),
        )
        if int(cfg.get("eval_every") or 1) <= 0:
            continue
        if epoch % int(cfg.get("eval_every") or 1) != 0:
            continue
        score = evaluate(
            model,
            tokenizer,
            valid_loader,
            device,
            gen_max_len=int(cfg["gen_max_len"]),
            num_beams=int(cfg["num_beams"]),
        )
        tqdm.write(f"epoch {epoch} train_loss {train_loss:.4f} score {score:.4f}")
        if score > state.best_score:
            state.best_score = score
            state.best_epoch = epoch
            save_checkpoint(model, tokenizer, str(cfg["output_dir"]), str(cfg["run_name"]), tag="best")
        if not bool(cfg.get("save_best_only", True)):
            save_checkpoint(model, tokenizer, str(cfg["output_dir"]), str(cfg["run_name"]), tag=f"epoch_{epoch}")

    if test_loader is not None and splits.test is not None:
        output_path = os.path.join(str(cfg["output_dir"]), str(cfg["run_name"]), "submission.csv")
        ids = splits.test[str(cfg["id_col"])].tolist()
        predict_to_csv(
            model,
            tokenizer,
            test_loader,
            ids,
            device=device,
            gen_max_len=int(cfg["gen_max_len"]),
            num_beams=int(cfg["num_beams"]),
            output_path=output_path,
        )

    return dict(
        best_score=state.best_score,
        best_epoch=state.best_epoch,
    )
