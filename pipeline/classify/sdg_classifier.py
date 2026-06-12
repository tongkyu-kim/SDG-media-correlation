"""
SDG Classifier for Korean news articles.

Pipeline:
  1. Korean keyword pre-screen  — fast catch of obvious matches
  2. Korean → English translation  (Helsinki-NLP/opus-mt-tc-big-ko-en)
  3. SDG classification on translated text  (jonas/sdg_classifier_osdg)

Using a translation step rather than a multilingual SDG model gives better
accuracy: the English OSDG model was trained on a large labelled dataset while
multilingual SDG fine-tunes are scarce and often undertrained for Korean.

Intensity mapping (0–3):
  final_score < 0.25  →  0  (not relevant)
  0.25 – 0.45         →  1  (indirectly mentioned)
  0.45 – 0.65         →  2  (moderately related)
  ≥ 0.65              →  3  (core focus)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

import torch

from classify.keywords_ko import keyword_scores

logger = logging.getLogger(__name__)

TRANSLATE_MODEL = "Helsinki-NLP/opus-mt-tc-big-ko-en"
SDG_MODEL       = "jonas/sdg_classifier_osdg"
MAX_TOKENS      = 512

KEYWORD_BOOST_PER_HIT = 0.04
MAX_KEYWORD_BOOST     = 0.20


@dataclass
class SDGResult:
    sdg_label:    int         # primary SDG 1–17, 0 = not classified
    sdg_score:    float       # fused confidence 0.0–1.0
    sdg_intensity: int        # 0–3
    all_scores:   dict = field(default_factory=dict)


def _score_to_intensity(score: float) -> int:
    if score < 0.25: return 0
    if score < 0.45: return 1
    if score < 0.65: return 2
    return 3


class SDGClassifier:
    """
    Korean → English translation + OSDG BERT classifier + keyword boost.
    Both models are lazy-loaded on first use.
    """

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._translator = None
        self._classifier = None
        self._label_map: dict[str, int] = {}

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_translator(self) -> None:
        if self._translator is not None:
            return
        from transformers import MarianMTModel, MarianTokenizer
        logger.info("Loading translation model %s ...", TRANSLATE_MODEL)
        self._tok = MarianTokenizer.from_pretrained(TRANSLATE_MODEL)
        self._mt  = MarianMTModel.from_pretrained(TRANSLATE_MODEL)
        if self.device == "cuda":
            self._mt = self._mt.cuda()
        self._mt.eval()
        self._translator = True   # sentinel: models are loaded
        logger.info("Translation model ready.")

    def _load_classifier(self) -> None:
        if self._classifier is not None:
            return
        from transformers import pipeline as hf_pipeline
        logger.info("Loading SDG model %s ...", SDG_MODEL)
        self._classifier = hf_pipeline(
            "text-classification",
            model=SDG_MODEL,
            tokenizer=SDG_MODEL,
            top_k=None,
            device=0 if self.device == "cuda" else -1,
            truncation=True,
            max_length=MAX_TOKENS,
        )
        # Build label → SDG int map
        id2label: dict = self._classifier.model.config.id2label
        for idx, lbl in id2label.items():
            m = re.search(r"(\d+)", str(lbl))
            if m:
                key = f"LABEL_{idx}"
                self._label_map[key] = int(m.group(1))
                self._label_map[lbl] = int(m.group(1))
        logger.info("SDG model ready. %d labels mapped.", len(self._label_map))

    # ── Translation ───────────────────────────────────────────────────────────

    def _translate(self, texts: List[str]) -> List[str]:
        """Translate a batch of Korean texts to English via MarianMT."""
        import torch
        inputs = self._tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=400,
        )
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            translated = self._mt.generate(**inputs)
        return [self._tok.decode(t, skip_special_tokens=True) for t in translated]

    # ── Fusion ────────────────────────────────────────────────────────────────

    def _fuse(self, model_scores: dict[int, float], original_ko: str) -> dict[int, float]:
        """Boost model scores using Korean keyword matches on original text."""
        kw = keyword_scores(original_ko)
        fused = dict(model_scores)
        for sdg, hits in kw.items():
            boost = min(hits * KEYWORD_BOOST_PER_HIT, MAX_KEYWORD_BOOST)
            fused[sdg] = min(1.0, fused.get(sdg, 0.0) + boost)
        return fused

    def _raw_to_result(self, raw: list, original_ko: str) -> SDGResult:
        model_scores: dict[int, float] = {}
        for item in raw:
            lbl = item["label"]
            sdg_num = self._label_map.get(lbl) or self._label_map.get(f"LABEL_{lbl}")
            if sdg_num:
                model_scores[sdg_num] = float(item["score"])

        fused = self._fuse(model_scores, original_ko)
        if not fused:
            return SDGResult(0, 0.0, 0)

        top = max(fused, key=fused.__getitem__)
        score = fused[top]
        return SDGResult(
            sdg_label    = top if score >= 0.25 else 0,
            sdg_score    = round(score, 4),
            sdg_intensity= _score_to_intensity(score),
            all_scores   = {k: round(v, 4) for k, v in fused.items()},
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def classify_batch(self, texts: List[str], batch_size: int = 32) -> List[SDGResult]:
        self._load_translator()
        self._load_classifier()

        results = []
        for i in range(0, len(texts), batch_size):
            batch_ko = [t[:600] for t in texts[i : i + batch_size]]

            # Translate Korean → English
            batch_en = self._translate(batch_ko)

            # Classify translated text
            raw_batch = self._classifier(batch_en)

            for ko_text, raw in zip(batch_ko, raw_batch):
                results.append(self._raw_to_result(raw, ko_text))

        return results

    def classify(self, text: str) -> SDGResult:
        return self.classify_batch([text])[0]
