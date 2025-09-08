import random
import re
from typing import List

from textblob import Word
from itertools import permutations

try:
    from transformers import pipeline
except Exception:  # pragma: no cover
    pipeline = None  # type: ignore


WORD_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return WORD_PATTERN.findall(text)


def detokenize(tokens: List[str]) -> str:
    text = ""
    for i, tok in enumerate(tokens):
        if i > 0 and re.match(r"\w", tok) and re.match(r"\w", tokens[i - 1]):
            text += " "
        text += tok
    return text


def synonym_replacement(text: str, n: int = 1, seed: int | None = None) -> str:
    rnd = random.Random(seed)
    tokens = tokenize(text)
    candidate_indices = [i for i, t in enumerate(tokens) if t.isalpha()]
    rnd.shuffle(candidate_indices)
    replaced = 0
    for idx in candidate_indices:
        word = tokens[idx]
        syns = Word(word).synsets
        lemmas = set(l.name().replace("_", " ") for s in syns for l in s.lemmas())
        lemmas.discard(word)
        lemmas = [w for w in lemmas if w.isalpha()]
        if not lemmas:
            continue
        tokens[idx] = rnd.choice(list(lemmas))
        replaced += 1
        if replaced >= n:
            break
    return detokenize(tokens)


def random_insertion(text: str, n: int = 1, seed: int | None = None) -> str:
    rnd = random.Random(seed)
    tokens = tokenize(text)
    words = [t for t in tokens if t.isalpha()]
    if not words:
        return text
    for _ in range(n):
        word = rnd.choice(words)
        pos = rnd.randint(0, len(tokens))
        tokens.insert(pos, word)
    return detokenize(tokens)


def random_swap(text: str, n: int = 1, seed: int | None = None) -> str:
    rnd = random.Random(seed)
    tokens = tokenize(text)
    indices = [i for i, t in enumerate(tokens) if t.isalpha()]
    for _ in range(n):
        if len(indices) < 2:
            break
        i, j = rnd.sample(indices, 2)
        tokens[i], tokens[j] = tokens[j], tokens[i]
    return detokenize(tokens)


def random_deletion(text: str, p: float = 0.1, seed: int | None = None) -> str:
    rnd = random.Random(seed)
    tokens = tokenize(text)
    kept: List[str] = []
    for t in tokens:
        if t.isalpha() and rnd.random() < p:
            continue
        kept.append(t)
    return detokenize(kept) if kept else text


def truncate(text: str, max_tokens: int = 30) -> str:
    tokens = tokenize(text)
    return detokenize(tokens[:max_tokens])


def noise_injection(text: str, n: int = 1, seed: int | None = None) -> str:
    rnd = random.Random(seed)
    tokens = tokenize(text)
    alpha_indices = [i for i, t in enumerate(tokens) if t.isalpha()]
    for _ in range(n):
        if not alpha_indices:
            break
        idx = rnd.choice(alpha_indices)
        word = list(tokens[idx])
        if len(word) >= 2:
            i = rnd.randrange(len(word))
            word[i] = rnd.choice("abcdefghijklmnopqrstuvwxyz")
            tokens[idx] = "".join(word)
    return detokenize(tokens)


def sentence_shuffle(text: str, seed: int | None = None) -> str:
    rnd = random.Random(seed)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) <= 1:
        return text
    rnd.shuffle(sentences)
    return " ".join(sentences)


def _get_pipeline(task: str, model: str | None = None):
    if pipeline is None:
        raise RuntimeError("transformers not installed")
    return pipeline(task, model=model)


def back_translation(text: str, src_to_mid: str = "Helsinki-NLP/opus-mt-en-de", mid_to_src: str = "Helsinki-NLP/opus-mt-de-en") -> str:
    try:
        trans_en2x = _get_pipeline("translation", model=src_to_mid)
        trans_x2en = _get_pipeline("translation", model=mid_to_src)
        mid = trans_en2x(text)[0]["translation_text"]
        back = trans_x2en(mid)[0]["translation_text"]
        return back
    except Exception:
        return text


def paraphrase(text: str, model_name: str = "t5-small", num_return_sequences: int = 1, max_length: int = 128, seed: int | None = None) -> str:
    try:
        gen = _get_pipeline("text2text-generation", model=model_name)
        prompt = f"paraphrase: {text}"
        outputs = gen(prompt, num_return_sequences=num_return_sequences, max_length=max_length, do_sample=True, temperature=0.9)
        return outputs[0]["generated_text"].strip()
    except Exception:
        return text

