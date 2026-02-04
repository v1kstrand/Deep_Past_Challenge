from __future__ import annotations

import re
from typing import List

import pandas as pd


class OptimizedPreprocessor:
    """Preprocessor with pre-compiled patterns."""

    def __init__(self):
        self.patterns = {
            "big_gap": re.compile(r"(\.{3,}|…+|……)"),
            "small_gap": re.compile(r"(xx+|\s+x\s+)"),
        }

    def preprocess_input_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        text = str(text)
        text = self.patterns["big_gap"].sub("<big_gap>", text)
        text = self.patterns["small_gap"].sub("<gap>", text)
        return text

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        s = pd.Series(texts).fillna("")
        s = s.astype(str)
        s = s.str.replace(self.patterns["big_gap"], "<big_gap>", regex=True)
        s = s.str.replace(self.patterns["small_gap"], "<gap>", regex=True)
        return s.tolist()


class VectorizedPostprocessor:
    """Vectorized postprocessing; aggressive mode is tunable."""

    def __init__(self, aggressive: bool = True):
        self.aggressive = aggressive
        self.patterns = {
            "gap": re.compile(r"(\[x\]|\(x\)|\bx\b)", re.I),
            "big_gap": re.compile(r"(\.{3,}|…|\[\.+\])"),
            "annotations": re.compile(
                r"\((fem|plur|pl|sing|singular|plural|\?|!)\..\s*\w*\)",
                re.I,
            ),
            "repeated_words": re.compile(r"\b(\w+)(?:\s+\1\b)+"),
            "whitespace": re.compile(r"\s+"),
            "punct_space": re.compile(r"\s+([.,:])"),
            "repeated_punct": re.compile(r"([.,])\1+"),
        }
        self.subscript_trans = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
        self.special_chars_trans = str.maketrans("ḫḪ", "hH")
        self.forbidden_chars = '!?()"——<>⌈⌋⌊[]+ʾ/;'
        self.forbidden_trans = str.maketrans("", "", self.forbidden_chars)

    def postprocess_batch(self, translations: List[str]) -> List[str]:
        s = pd.Series(translations)
        valid_mask = s.apply(lambda x: isinstance(x, str) and x.strip())
        if not valid_mask.all():
            s[~valid_mask] = ""

        s = s.str.translate(self.special_chars_trans)
        s = s.str.translate(self.subscript_trans)
        s = s.str.replace(self.patterns["whitespace"], " ", regex=True)
        s = s.str.strip()

        if self.aggressive:
            s = s.str.replace(self.patterns["gap"], "<gap>", regex=True)
            s = s.str.replace(self.patterns["big_gap"], "<big_gap>", regex=True)
            s = s.str.replace("<gap> <gap>", "<big_gap>", regex=False)
            s = s.str.replace("<big_gap> <big_gap>", "<big_gap>", regex=False)
            s = s.str.replace(self.patterns["annotations"], "", regex=True)

            s = s.str.replace("<gap>", "\x00GAP\x00", regex=False)
            s = s.str.replace("<big_gap>", "\x00BIG\x00", regex=False)
            s = s.str.translate(self.forbidden_trans)
            s = s.str.replace("\x00GAP\x00", " <gap> ", regex=False)
            s = s.str.replace("\x00BIG\x00", " <big_gap> ", regex=False)

            s = s.str.replace(r"(\d+)\.5\b", r"\1½", regex=True)
            s = s.str.replace(r"\b0\.5\b", "½", regex=True)
            s = s.str.replace(r"(\d+)\.25\b", r"\1¼", regex=True)
            s = s.str.replace(r"\b0\.25\b", "¼", regex=True)
            s = s.str.replace(r"(\d+)\.75\b", r"\1¾", regex=True)
            s = s.str.replace(r"\b0\.75\b", "¾", regex=True)

            s = s.str.replace(self.patterns["repeated_words"], r"\1", regex=True)
            for n in range(4, 1, -1):
                pattern = (
                    r"\b((?:\w+\s+){" + str(n - 1) + r"}\w+)(?:\s+\1\b)+"
                )
                s = s.str.replace(pattern, r"\1", regex=True)

            s = s.str.replace(self.patterns["punct_space"], r"\1", regex=True)
            s = s.str.replace(self.patterns["repeated_punct"], r"\1", regex=True)
            s = s.str.replace(self.patterns["whitespace"], " ", regex=True)
            s = s.str.strip().str.strip("-").str.strip()

        return s.tolist()
