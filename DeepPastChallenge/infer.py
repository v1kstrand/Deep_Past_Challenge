from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from .textproc import OptimizedPreprocessor, VectorizedPostprocessor
from .utils import filter_kwargs


def apply_env_defaults() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    os.environ.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


def setup_logging(output_dir: str) -> logging.Logger:
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    log_file = Path(output_dir) / "inference_ultra.log"

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    return logging.getLogger(__name__)


@dataclass
class UltraConfig:
    test_data_path: str = ""
    data_root: str | None = None
    test_path: str = "test.csv"
    model_path: str = ""
    output_dir: str = "./outputs"
    id_col: str = "id"
    src_col: str = "transliteration"
    src_prefix: str = "translate Akkadian to English: "

    max_length: int = 512
    batch_size: int = 8
    num_workers: int = 4

    num_beams: int = 8
    max_new_tokens: int = 512
    length_penalty: float = 1.5
    repetition_penalty: float = 1.2
    early_stopping: bool = True
    no_repeat_ngram_size: int = 0

    use_mixed_precision: bool = True
    precision: str = "bf16"
    use_better_transformer: bool = True
    use_bucket_batching: bool = True
    use_vectorized_postproc: bool = True
    use_adaptive_beams: bool = True
    use_auto_batch_size: bool = False

    aggressive_postprocessing: bool = True
    checkpoint_freq: int = 100
    num_buckets: int = 4

    def resolve_test_path(self) -> str:
        if self.test_data_path:
            return self.test_data_path
        if self.data_root:
            return str(Path(self.data_root) / self.test_path)
        return self.test_path

    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_precision(precision: str) -> torch.dtype:
    s = str(precision or "").strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size: int, num_buckets: int = 4, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)

        lengths = [len(text.split()) for _, text in dataset]
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

        bucket_size = max(1, len(sorted_indices) // int(num_buckets))
        self.buckets = []
        for i in range(int(num_buckets)):
            start = i * bucket_size
            end = None if i == int(num_buckets) - 1 else (i + 1) * bucket_size
            self.buckets.append(sorted_indices[start:end])

    def __iter__(self):
        import random

        for bucket in self.buckets:
            if self.shuffle:
                random.shuffle(bucket)
            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i : i + self.batch_size]

    def __len__(self):
        return sum((len(b) + self.batch_size - 1) // self.batch_size for b in self.buckets)


class AkkadianDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, preprocessor: OptimizedPreprocessor, *, id_col: str, src_col: str, src_prefix: str):
        self.sample_ids = dataframe[id_col].tolist()
        raw_texts = dataframe[src_col].tolist()
        preprocessed = preprocessor.preprocess_batch(raw_texts)
        self.input_texts = [f"{src_prefix}{text}" for text in preprocessed]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index: int):
        return self.sample_ids[index], self.input_texts[index]


class UltraInferenceEngine:
    def __init__(self, config: UltraConfig):
        self.config = config
        self.logger = setup_logging(self.config.output_dir)
        self.preprocessor = OptimizedPreprocessor()
        self.postprocessor = VectorizedPostprocessor(aggressive=config.aggressive_postprocessing)
        self.results: list[tuple[Any, str]] = []
        self._load_model()

    def _load_model(self):
        self.logger.info(f"Loading model from {self.config.model_path}")
        config = AutoConfig.from_pretrained(self.config.model_path)
        config.tie_word_embeddings = False
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_path,
            config=config,
        ).to(self.config.device()).eval()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, use_fast=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, use_fast=False)

        if self.config.use_better_transformer and torch.cuda.is_available():
            try:
                from optimum.bettertransformer import BetterTransformer

                self.logger.info("Applying BetterTransformer...")
                self.model = BetterTransformer.transform(self.model)
                self.logger.info("BetterTransformer applied")
            except ImportError:
                self.logger.warning("optimum not installed; skipping BetterTransformer")
            except Exception as exc:
                self.logger.warning(f"BetterTransformer failed: {exc}")

    def _collate_fn(self, batch_samples):
        batch_ids = [s[0] for s in batch_samples]
        batch_texts = [s[1] for s in batch_samples]
        tokenized = self.tokenizer(
            batch_texts,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return batch_ids, tokenized

    def find_optimal_batch_size(self, dataset, start_bs: int = 32):
        self.logger.info("Finding optimal batch size...")
        max_bs = int(start_bs)
        min_bs = 1
        while max_bs - min_bs > 1:
            test_bs = (max_bs + min_bs) // 2
            try:
                test_batch = [dataset[i] for i in range(min(test_bs, len(dataset)))]
                _, inputs = self._collate_fn(test_batch)
                with torch.inference_mode():
                    outputs = self._generate(inputs, max_new_tokens=64)
                min_bs = test_bs
                del outputs, inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    max_bs = test_bs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise
        self.logger.info(f"Optimal batch size: {min_bs}")
        return min_bs

    def _get_adaptive_beam_size(self, attention_mask: torch.Tensor) -> int:
        if not self.config.use_adaptive_beams:
            return int(self.config.num_beams)
        lengths = attention_mask.sum(dim=1)
        short_beams = max(4, int(self.config.num_beams) // 2)
        beam_sizes = torch.where(lengths < 100, torch.tensor(short_beams), torch.tensor(self.config.num_beams))
        return int(beam_sizes[0].item())

    def _save_checkpoint(self):
        if self.config.checkpoint_freq <= 0:
            return
        if len(self.results) > 0 and len(self.results) % int(self.config.checkpoint_freq) == 0:
            path = Path(self.config.output_dir) / f"checkpoint_{len(self.results)}.csv"
            df = pd.DataFrame(self.results, columns=["id", "translation"])
            df.to_csv(path, index=False)
            self.logger.info(f"Checkpoint saved: {len(self.results)} translations")

    def _generate(self, tokenized, *, max_new_tokens: Optional[int] = None):
        input_ids = tokenized.input_ids.to(self.config.device())
        attention_mask = tokenized.attention_mask.to(self.config.device())
        beam_size = self._get_adaptive_beam_size(attention_mask)
        gen_config = {
            "max_new_tokens": int(max_new_tokens or self.config.max_new_tokens),
            "length_penalty": float(self.config.length_penalty),
            "repetition_penalty": float(self.config.repetition_penalty),
            "early_stopping": bool(self.config.early_stopping),
            "use_cache": True,
            "num_beams": int(beam_size),
        }
        if int(self.config.no_repeat_ngram_size) > 0:
            gen_config["no_repeat_ngram_size"] = int(self.config.no_repeat_ngram_size)

        if self.config.use_mixed_precision and torch.cuda.is_available():
            dtype = _parse_precision(self.config.precision)
            if dtype == torch.float32:
                return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_config)
            with autocast(dtype=dtype):
                return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_config)
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_config)

    def run_inference(self, test_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Starting inference")
        dataset = AkkadianDataset(
            test_df,
            self.preprocessor,
            id_col=self.config.id_col,
            src_col=self.config.src_col,
            src_prefix=self.config.src_prefix,
        )

        if self.config.use_auto_batch_size:
            self.config.batch_size = self.find_optimal_batch_size(dataset)

        if self.config.use_bucket_batching:
            batch_sampler = BucketBatchSampler(
                dataset,
                self.config.batch_size,
                num_buckets=self.config.num_buckets,
            )
            dataloader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=self.config.num_workers,
                collate_fn=self._collate_fn,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True if self.config.num_workers > 0 else False,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=self._collate_fn,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True if self.config.num_workers > 0 else False,
            )

        self.results = []
        with torch.inference_mode():
            for batch_idx, (batch_ids, tokenized) in enumerate(tqdm(dataloader, desc="Translating")):
                try:
                    outputs = self._generate(tokenized)
                    translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    if self.config.use_vectorized_postproc:
                        cleaned = self.postprocessor.postprocess_batch(translations)
                    else:
                        cleaned = translations
                    self.results.extend(zip(batch_ids, cleaned))
                    self._save_checkpoint()
                    if torch.cuda.is_available() and batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                except Exception as exc:
                    self.logger.error(f"Batch {batch_idx} failed: {exc}")
                    self.results.extend([(bid, "") for bid in batch_ids])

        results_df = pd.DataFrame(self.results, columns=["id", "translation"])
        self._validate_results(results_df)
        return results_df

    def _validate_results(self, df: pd.DataFrame):
        empty = df["translation"].str.strip().eq("").sum()
        lengths = df["translation"].str.len()
        self.logger.info(f"Empty translations: {empty} ({empty/len(df)*100:.2f}%)")
        self.logger.info(
            f"Length stats: mean={lengths.mean():.1f} median={lengths.median():.1f} "
            f"min={lengths.min()} max={lengths.max()}"
        )


def run_inference_from_config(cfg: dict[str, Any]) -> pd.DataFrame:
    apply_env_defaults()
    config = UltraConfig(**filter_kwargs(UltraConfig, cfg))
    engine = UltraInferenceEngine(config)
    test_df = pd.read_csv(config.resolve_test_path(), encoding="utf-8")
    results_df = engine.run_inference(test_df)
    output_path = Path(config.output_dir) / "submission.csv"
    results_df.to_csv(output_path, index=False)
    return results_df
