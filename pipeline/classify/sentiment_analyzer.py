"""
Korean news sentiment analyzer.

Model: FISA-conclave/klue-roberta-news-sentiment
  - Fine-tuned KLUE-RoBERTa on Korean news data
  - Labels: positive / neutral / negative (exact label strings depend on
    the model's config — we normalize to our three-class scheme below)

Output label: "positive" | "neutral" | "negative"
Output score: confidence 0.0-1.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import torch

logger = logging.getLogger(__name__)

MODEL_ID = "FISA-conclave/klue-roberta-news-sentiment"
MAX_INPUT_TOKENS = 512

# Normalize whatever labels the model returns to our three-class scheme.
# Keys are lower-case model label strings; values are our canonical labels.
_LABEL_NORM = {
    # Common positive variants
    "positive": "positive",
    "pos":      "positive",
    "긍정":     "positive",
    "호의적":   "positive",
    # Common negative variants
    "negative": "negative",
    "neg":      "negative",
    "부정":     "negative",
    "비판적":   "negative",
    # Neutral / mixed
    "neutral":  "neutral",
    "neu":      "neutral",
    "중립":     "neutral",
    "mixed":    "neutral",
}


@dataclass
class SentimentResult:
    label: str          # "positive" | "neutral" | "negative"
    score: float        # confidence for the predicted label


def _normalize_label(raw: str) -> str:
    return _LABEL_NORM.get(raw.lower(), "neutral")


class SentimentAnalyzer:
    """
    Wraps the KLUE-RoBERTa news sentiment model.
    Lazy-loads on first call.
    """

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        from transformers import pipeline as hf_pipeline
        logger.info("Loading sentiment model %s on %s ...", self.model_id, self.device)
        self._pipeline = hf_pipeline(
            "text-classification",
            model=self.model_id,
            tokenizer=self.model_id,
            top_k=1,
            device=0 if self.device == "cuda" else -1,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )
        logger.info("Sentiment model ready.")

    def analyze(self, text: str) -> SentimentResult:
        self._load()
        raw = self._pipeline(text[:1500])[0]
        top = raw[0] if isinstance(raw, list) else raw
        return SentimentResult(
            label=_normalize_label(top["label"]),
            score=round(float(top["score"]), 4),
        )

    def analyze_batch(self, texts: List[str], batch_size: int = 64) -> List[SentimentResult]:
        self._load()
        results = []
        for i in range(0, len(texts), batch_size):
            batch = [t[:1500] for t in texts[i : i + batch_size]]
            raw_batch = self._pipeline(batch)
            for raw in raw_batch:
                top = raw[0] if isinstance(raw, list) else raw
                results.append(SentimentResult(
                    label=_normalize_label(top["label"]),
                    score=round(float(top["score"]), 4),
                ))
        return results
