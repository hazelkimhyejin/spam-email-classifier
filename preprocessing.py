"""Shared preprocessing so training and the app clean text identically."""
import re


def clean_text(text: str) -> str:
    """Lowercase, normalise URLs/numbers, strip punctuation, collapse spaces."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
