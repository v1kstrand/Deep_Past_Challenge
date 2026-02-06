from __future__ import annotations

import math
from typing import Sequence

import sacrebleu


def _refs(references: Sequence[str]) -> list[list[str]]:
    # sacrebleu corpus_* expects list-of-references (list of lists)
    return [list(references)]


def bleu_score(predictions: Sequence[str], references: Sequence[str]) -> float:
    return float(sacrebleu.corpus_bleu(list(predictions), _refs(references)).score)


def chrfpp_score(predictions: Sequence[str], references: Sequence[str]) -> float:
    # Official metric notebook uses chrF++ with word_order=2.
    return float(
        sacrebleu.corpus_chrf(list(predictions), _refs(references), word_order=2).score
    )


def geometric_mean_bleu_chrfpp(bleu: float, chrfpp: float) -> float:
    # Scores are 0-100; GM stays on 0-100.
    return math.sqrt(max(bleu, 0.0) * max(chrfpp, 0.0))


def kaggle_scores(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """
    Official Kaggle metric for this competition:
    GM = sqrt(BLEU * chrF++) where BLEU and chrF++ are micro-averaged over the corpus.
    """
    bleu = bleu_score(predictions, references)
    chrfpp = chrfpp_score(predictions, references)
    gm = geometric_mean_bleu_chrfpp(bleu, chrfpp)
    return {"bleu": bleu, "chrfpp": chrfpp, "gm": gm}
