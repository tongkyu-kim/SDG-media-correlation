"""
Korean news sentiment analyzer.

Model: FISA-conclave/klue-roberta-news-sentiment
  - Fine-tuned KLUE-RoBERTa on Korean news data
  - Labels: positive / neutral / negative

Output label:      "positive" | "neutral" | "negative"
Output score:      confidence of winning class (0.0–1.0)
Output continuous: P(positive) − P(negative)  ∈ [−1, +1]
  Used as sentiment_score in the panel (H3 sentiment heterogeneity).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import torch

logger = logging.getLogger(__name__)

MODEL_ID         = "FISA-conclave/klue-roberta-news-sentiment"
MAX_INPUT_TOKENS = 512

_LABEL_NORM = {
    "positive": "positive", "pos": "positive", "긍정": "positive", "호의적": "positive",
    "negative": "negative", "neg": "negative", "부정": "negative", "비판적": "negative",
    "neutral":  "neutral",  "neu": "neutral",  "중립": "neutral",  "mixed":  "neutral",
}


@dataclass
class SentimentResult:
    label:      str    # "positive" | "neutral" | "negative"
    score:      float  # confidence of the winning class (0.0–1.0)
    continuous: float  # P(positive) − P(negative) ∈ [−1, +1]


def _normalize_label(raw: str) -> str:
    return _LABEL_NORM.get(raw.lower(), "neutral")


def _probs_to_result(all_preds: list[dict]) -> SentimentResult:
    """
    Convert the full list of {label, score} dicts (one per class) into a
    SentimentResult.  Works whether top_k=None returns 2 or 3 classes.
    """
    prob: dict[str, float] = {}
    for p in all_preds:
        norm = _normalize_label(p["label"])
        prob[norm] = prob.get(norm, 0.0) + p["score"]

    p_pos = prob.get("positive", 0.0)
    p_neg = prob.get("negative", 0.0)
    top   = max(prob, key=prob.__getitem__)

    return SentimentResult(
        label=top,
        score=round(prob[top], 4),
        continuous=round(p_pos - p_neg, 4),
    )


class SentimentAnalyzer:
    """
    Wraps the KLUE-RoBERTa news sentiment model.
    Lazy-loads on first call.
    Uses top_k=None to retrieve all class probabilities, enabling
    the continuous P(pos)−P(neg) score without any retraining.
    """

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        self.model_id = model_id
        self.device   = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
            top_k=None,          # get all class probabilities
            device=0 if self.device == "cuda" else -1,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )
        logger.info("Sentiment model ready.")

    def analyze(self, text: str) -> SentimentResult:
        self._load()
        raw = self._pipeline(text[:1500])[0]
        return _probs_to_result(raw if isinstance(raw, list) else [raw])

    def analyze_batch(self, texts: List[str], batch_size: int = 64) -> List[SentimentResult]:
        self._load()
        results = []
        for i in range(0, len(texts), batch_size):
            batch     = [t[:1500] for t in texts[i : i + batch_size]]
            raw_batch = self._pipeline(batch)
            for raw in raw_batch:
                results.append(_probs_to_result(raw if isinstance(raw, list) else [raw]))
        return results
