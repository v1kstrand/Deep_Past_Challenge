from __future__ import annotations

import inspect
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers.trainer_callback import ProgressCallback
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate

from .config import DEFAULTS
from .comet_utils import CometHFCallback, maybe_init_comet
from .tqdm_callback import DPCProgressCallback
from .textproc import OptimizedPreprocessor


def _training_args_strategy_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    `transformers` renamed `evaluation_strategy` -> `eval_strategy` in newer versions.
    Select the supported kwarg at runtime so we work across notebook images.
    """
    params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    eval_val = str(cfg.get("eval_strategy") or "epoch")
    save_val = str(cfg.get("save_strategy") or "epoch")

    out: dict[str, Any] = {}
    if "evaluation_strategy" in params:
        out["evaluation_strategy"] = eval_val
    elif "eval_strategy" in params:
        out["eval_strategy"] = eval_val
    else:
        raise TypeError("Seq2SeqTrainingArguments does not support eval strategy argument")

    # As of today, `save_strategy` exists in both old and new versions, but guard anyway.
    if "save_strategy" in params:
        out["save_strategy"] = save_val

    return out


def simple_sentence_aligner(df: pd.DataFrame, *, min_len: int = 3) -> pd.DataFrame:
    aligned_data: list[dict[str, str]] = []
    for _, row in df.iterrows():
        src = str(row["transliteration"])
        tgt = str(row["translation"])

        tgt_sents = [t.strip() for t in re.split(r"(?<=[.!?])\s+", tgt) if t.strip()]
        src_lines = [s.strip() for s in src.split("\n") if s.strip()]

        if len(tgt_sents) > 1 and len(tgt_sents) == len(src_lines):
            for s, t in zip(src_lines, tgt_sents):
                if len(s) > min_len and len(t) > min_len:
                    aligned_data.append({"transliteration": s, "translation": t})
        else:
            aligned_data.append({"transliteration": src, "translation": tgt})
    return pd.DataFrame(aligned_data)


def _resolve_path(root: str | None, path: str) -> str:
    if root is None:
        return path
    if path.startswith(("~", "/", "\\")):
        return path
    return os.path.join(root, path)


def _dtype_flags(trainable_dtype: str | None) -> tuple[bool, bool]:
    s = str(trainable_dtype or "").strip().lower()
    if s in ("bf16", "bfloat16"):
        return False, True
    if s in ("fp16", "float16", "half"):
        return True, False
    return False, False


def run_training_trainer(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg: dict[str, Any] = dict(DEFAULTS)
    if overrides:
        cfg.update(overrides)

    exp = maybe_init_comet(cfg)
    data_root = cfg.get("data_root")
    train_path = _resolve_path(data_root, str(cfg["train_path"]))
    df = pd.read_csv(train_path)

    if bool(cfg.get("use_sentence_aligner", False)):
        min_len = int(cfg.get("align_min_len", 3))
        df = simple_sentence_aligner(df, min_len=min_len)

    dataset = Dataset.from_pandas(df)
    split = dataset.train_test_split(test_size=float(cfg["val_split"]), seed=int(cfg["seed"]))

    model_name = str(cfg["model_name"])
    tokenizer_name = str(cfg.get("tokenizer_name") or model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    preprocessor = OptimizedPreprocessor() if bool(cfg.get("preprocess_inputs", True)) else None
    src_prefix = str(cfg.get("src_prefix") or "")
    src_col = str(cfg.get("src_col", "transliteration"))
    tgt_col = str(cfg.get("tgt_col", "translation"))
    max_source_len = int(cfg.get("max_source_len") or 512)
    max_target_len = int(cfg.get("max_target_len") or 512)

    def preprocess_function(examples):
        inputs = [str(ex) for ex in examples[src_col]]
        if preprocessor is not None:
            inputs = preprocessor.preprocess_batch(inputs)
        inputs = [src_prefix + ex for ex in inputs]
        targets = [str(ex) for ex in examples[tgt_col]]
        model_inputs = tokenizer(inputs, max_length=max_source_len, truncation=True)
        labels = tokenizer(targets, max_length=max_target_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = split["train"].map(preprocess_function, batched=True)
    tokenized_val = split["test"].map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    metric = evaluate.load("chrf")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"chrf": result["score"]}

    output_dir = str(cfg.get("output_dir") or "./outputs")
    run_name = str(cfg.get("run_name") or "trainer_baseline")
    output_dir = os.path.join(output_dir, run_name)

    fp16, bf16 = _dtype_flags(cfg.get("trainable_dtype"))
    eval_max_len = int(cfg.get("val_gen_max_len") or cfg.get("gen_max_len") or max_target_len)
    eval_beams = int(cfg.get("val_num_beams") or cfg.get("num_beams") or 4)

    strategy_kwargs = _training_args_strategy_kwargs(cfg)
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        **strategy_kwargs,
        learning_rate=float(cfg.get("lr") or 2e-4),
        per_device_train_batch_size=int(cfg.get("batch_size") or 8),
        per_device_eval_batch_size=int(cfg.get("eval_batch_size") or cfg.get("batch_size") or 8),
        num_train_epochs=float(cfg.get("epochs") or 3),
        predict_with_generate=True,
        generation_max_length=eval_max_len,
        generation_num_beams=eval_beams,
        logging_steps=int(cfg.get("log_every") or 50),
        save_total_limit=int(cfg.get("save_total_limit") or 2),
        fp16=fp16,
        bf16=bf16,
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
    )

    if cfg.get("gradient_accumulation_steps"):
        args.gradient_accumulation_steps = int(cfg.get("gradient_accumulation_steps"))

    callbacks = []
    if exp is not None:
        callbacks.append(CometHFCallback(exp))

    # `tokenizer` is deprecated in newer transformers in favor of `processing_class`.
    trainer_init_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    trainer_tok_kwargs: dict[str, Any] = {}
    if "processing_class" in trainer_init_params:
        trainer_tok_kwargs["processing_class"] = tokenizer
    else:
        trainer_tok_kwargs["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        **trainer_tok_kwargs,
    )

    # Replace HF default progress callback so we can show extra fields in the bar postfix.
    try:
        trainer.remove_callback(ProgressCallback)
    except Exception:
        pass
    trainer.add_callback(DPCProgressCallback())

    try:
        train_result = trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        return {
            "train_runtime": train_result.metrics.get("train_runtime"),
            "train_samples": train_result.metrics.get("train_samples"),
        }
    finally:
        if exp is not None:
            try:
                exp.end()
            except Exception:
                pass
