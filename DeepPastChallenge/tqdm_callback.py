from __future__ import annotations

from typing import Any

from transformers.trainer_callback import ProgressCallback


def _pick(d: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in keys:
        if k not in d:
            continue
        v = d[k]
        if isinstance(v, float):
            out[k] = float(f"{v:.4f}")
        else:
            out[k] = v
    return out


class DPCProgressCallback(ProgressCallback):
    """
    Extend HF's tqdm progress bar to show useful live stats in the postfix.
    """

    def __init__(self):
        super().__init__()
        self._latest: dict[str, Any] = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Let HF handle its usual printing/progress updates first.
        super().on_log(args, state, control, logs=logs, **kwargs)
        if not logs:
            return

        # Keep common training keys.
        self._latest.update(
            _pick(
                logs,
                keys=[
                    "loss",
                    "learning_rate",
                    "grad_norm",
                    "epoch",
                ],
            )
        )

        if self.training_bar is not None and self._latest:
            try:
                self.training_bar.set_postfix(self._latest, refresh=False)
            except Exception:
                pass

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        super().on_evaluate(args, state, control, metrics=metrics, **kwargs)
        if not metrics:
            return

        # Show key eval metrics on the training bar too.
        self._latest.update(
            _pick(
                metrics,
                keys=[
                    "eval_loss",
                    "eval_chrf",
                    "eval_runtime",
                    "epoch",
                ],
            )
        )
        if self.training_bar is not None and self._latest:
            try:
                self.training_bar.set_postfix(self._latest, refresh=False)
            except Exception:
                pass

