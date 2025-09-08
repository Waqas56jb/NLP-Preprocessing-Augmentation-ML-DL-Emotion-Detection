import re
from typing import List, Callable, Dict

import nltk
from nltk.corpus import stopwords


def ensure_nltk_resources() -> None:
    """Download required NLTK resources if missing."""
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")


WHITESPACE_PATTERN = re.compile(r"\s+")
NUMBERS_PATTERN = re.compile(r"\d+")


def to_lowercase(text: str) -> str:
    return text.lower()


def remove_punctuation(text: str) -> str:
    # Remove underscores as well, then remove any non-alphanumeric, non-space chars
    text = text.replace("_", " ")
    return re.sub(r"[^A-Za-z0-9\s]", " ", text)


def remove_numbers(text: str) -> str:
    return NUMBERS_PATTERN.sub(" ", text)


def remove_extra_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def remove_stopwords(text: str, language: str = "english") -> str:
    ensure_nltk_resources()
    stop_words = set(stopwords.words(language))
    tokens = text.split()
    kept_tokens = [t for t in tokens if t.lower() not in stop_words]
    return " ".join(kept_tokens)


PREPROCESSORS: Dict[str, Callable[[str], str]] = {
    "lowercase": to_lowercase,
    "remove_punctuation": remove_punctuation,
    "remove_numbers": remove_numbers,
    "remove_extra_whitespace": remove_extra_whitespace,
}


def apply_pipeline(text: str, steps: List[str], remove_stop_words: bool = False) -> str:
    """Apply a sequence of preprocessing steps to text.

    steps: keys from PREPROCESSORS in order.
    remove_stop_words: if True, apply stopwords removal at the end.
    """
    processed = text
    for step in steps:
        func = PREPROCESSORS.get(step)
        if func is None:
            continue
        processed = func(processed)
    if remove_stop_words:
        processed = remove_stopwords(processed)
        processed = remove_extra_whitespace(processed)
    return processed

