from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd
import nltk
from nltk.corpus import stopwords


SUGGESTION_THRESHOLDS = {
    "punctuation_ratio": 0.05,
    "digit_ratio": 0.02,
    "avg_whitespace_run": 1.2,
    "stopword_share": 0.4,
}


@dataclass
class DatasetStats:
    punctuation_ratio: float
    digit_ratio: float
    avg_whitespace_run: float
    stopword_share: float


def _safe_len(x: List[str]) -> int:
    return len(x) if isinstance(x, list) else 0


def analyze_texts(texts: List[str]) -> DatasetStats:
    total_chars = 0
    punct_chars = 0
    digit_chars = 0
    whitespace_runs = []
    stopword_count = 0
    token_count = 0

    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")

    stop_words = set(stopwords.words("english"))
    for t in texts:
        if not isinstance(t, str):
            continue
        total_chars += len(t)
        punct_chars += sum(1 for ch in t if re.match(r"[^\w\s]", ch))
        digit_chars += sum(1 for ch in t if ch.isdigit())

        for run in re.findall(r"\s+", t):
            whitespace_runs.append(len(run))

        toks = re.findall(r"\w+", t)
        token_count += len(toks)
        stopword_count += sum(1 for w in toks if w.lower() in stop_words)

    punctuation_ratio = (punct_chars / total_chars) if total_chars else 0.0
    digit_ratio = (digit_chars / total_chars) if total_chars else 0.0
    avg_whitespace_run = (sum(whitespace_runs) / len(whitespace_runs)) if whitespace_runs else 1.0
    stopword_share = (stopword_count / token_count) if token_count else 0.0

    return DatasetStats(
        punctuation_ratio=punctuation_ratio,
        digit_ratio=digit_ratio,
        avg_whitespace_run=avg_whitespace_run,
        stopword_share=stopword_share,
    )


def suggest_preprocessing(stats: DatasetStats) -> Dict[str, bool]:
    return {
        "lowercase": True,
        "remove_punctuation": stats.punctuation_ratio > SUGGESTION_THRESHOLDS["punctuation_ratio"],
        "remove_numbers": stats.digit_ratio > SUGGESTION_THRESHOLDS["digit_ratio"],
        "remove_extra_whitespace": stats.avg_whitespace_run > SUGGESTION_THRESHOLDS["avg_whitespace_run"],
        "remove_stopwords": stats.stopword_share > SUGGESTION_THRESHOLDS["stopword_share"],
    }


def load_and_analyze(path: str, text_column: str = "text") -> tuple[pd.DataFrame, DatasetStats, Dict[str, bool]]:
    df = pd.read_csv(path)
    texts = df[text_column].astype(str).tolist()
    stats = analyze_texts(texts)
    suggestions = suggest_preprocessing(stats)
    return df, stats, suggestions


def class_imbalance_suggestion(df: pd.DataFrame, label_col: str = "Emotion", threshold_ratio: float = 0.5) -> bool:
    counts = df[label_col].value_counts(normalize=True)
    if counts.empty:
        return False
    largest = counts.iloc[0]
    return largest > threshold_ratio

