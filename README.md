# Deep Past Challenge

Machine translation baseline for the Deep Past Initiative (Akkadian transliteration → English).

## Install

```bash
pip install -e .
```

## Comet (Optional)

```bash
pip install -e '.[comet]'
export COMET_API_KEY='...'
```

Enable by setting `comet_project_name` in your YAML config. The Comet experiment name is set to `run_name`.

## Training

Run the training entrypoint from the repo root:

```bash
python DeepPastChallenge/scripts/train_cv_func.py --overrides /path/to/train.yaml
```

Notes:
- Source preprocessing is enabled by default (`preprocess_inputs=true`).
- Use `src_prefix` if your model expects a task prefix.
- Validation decode can be faster with `val_num_beams` and `val_gen_max_len`.
- Example config: `configs/train_base.yaml`

## Trainer Baseline

Trainer-based baseline aligned with the Kaggle starter notebook:

```bash
python DeepPastChallenge/scripts/train_trainer.py --overrides configs/train_trainer_base.yaml
```

Notes:
- Uses `datasets` + `evaluate` with chrF metric.
- Optional `use_sentence_aligner` can be toggled.

## Inference

The inference pipeline ports the community notebook optimizations (bucketed batching, adaptive beams, vectorized postprocessing).

```bash
python DeepPastChallenge/scripts/infer.py --overrides configs/infer_base.yaml
```

Notes:
- `aggressive_postprocessing` is tunable in the infer config.
- BetterTransformer is optional; install `optimum` to enable it.

## Layout

- `DeepPastChallenge/` current codebase
- `DeepPastChallenge/v0/` prior baseline for reference
