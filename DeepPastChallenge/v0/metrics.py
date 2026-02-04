from __future__ import annotations

import math
from typing import Iterable, Sequence

import sacrebleu


def compute_bleu(
    predictions: Sequence[str],
    references: Sequence[str],
) -> float:
    refs = [list(references)]
    bleu = sacrebleu.corpus_bleu(predictions, refs)
    return float(bleu.score)


def compute_chrf(
    predictions: Sequence[str],
    references: Sequence[str],
) -> float:
    refs = [list(references)]
    chrf = sacrebleu.corpus_chrf(predictions, refs)
    return float(chrf.score)


def kaggle_mt_score(
    predictions: Sequence[str],
    references: Sequence[str],
) -> float:
    """
    Kaggle metric: geometric mean of BLEU and chrF++ (scores in 0-100 range).
    """
    bleu = compute_bleu(predictions, references)
    chrf = compute_chrf(predictions, references)
    return math.sqrt(max(bleu, 0.0) * max(chrf, 0.0))
