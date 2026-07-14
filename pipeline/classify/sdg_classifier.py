"""
SDG Classifier for Korean news articles.

Two-step method:
  1. Helsinki-NLP/opus-mt-ko-en  — translate Korean text to English
  2. intfloat/multilingual-e5-base — embed translated English text
  3. Cosine similarity against English-only ODA/development-specific anchors

Why translation?  Multilingual-e5 cannot separate "domestic Korean health"
from "global health ODA" in the Korean embedding space — both are "health."
After translation to English, "erectile dysfunction drug" produces a very
different embedding from "malaria treatment in sub-Saharan Africa."

English-only development anchors (SDG_ANCHORS below) are explicitly framed
around ODA, developing countries, Africa/Asia/recipient countries so that
domestic Korean articles score below the relevance threshold even if they
mention an SDG topic in a domestic context.

Intensity mapping (0–3):
  similarity < 0.35  →  0  (not development-relevant)
  0.35 – 0.50        →  1  (indirectly mentioned)
  0.50 – 0.65        →  2  (moderately related)
  ≥ 0.65             →  3  (core focus)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import torch
import numpy as np

from classify.keywords_ko import keyword_scores

logger = logging.getLogger(__name__)

TRANSLATE_MODEL   = "Helsinki-NLP/opus-mt-ko-en"
EMBED_MODEL       = "intfloat/multilingual-e5-base"
SIM_THRESHOLD     = 0.35          # lower than before — English anchors score lower overall
KEYWORD_BOOST_PER_HIT = 0.03
MAX_KEYWORD_BOOST     = 0.15

# ── English-only ODA/development SDG anchor texts ────────────────────────────
# Anchors are explicitly framed as INTERNATIONAL DEVELOPMENT contexts.
# After Korean→English translation, domestic articles score far below threshold.

SDG_ANCHORS: dict[int, str] = {
    1: (
        "Extreme poverty reduction through ODA in developing countries. "
        "International aid for the poorest populations in Africa, Asia, Latin America. "
        "Social protection programs in least developed countries. "
        "Humanitarian assistance for vulnerable people. "
        "KOICA poverty alleviation projects in recipient countries."
    ),
    2: (
        "Hunger relief and food security in developing countries. "
        "Famine response and malnutrition programs in Africa and Asia. "
        "World Food Programme WFP food aid and emergency feeding. "
        "Agricultural development ODA in recipient countries. "
        "Nutrition support for children in least developed countries."
    ),
    3: (
        "International health aid in developing countries through ODA. "
        "Malaria HIV AIDS tuberculosis Ebola outbreak response in Africa and Asia. "
        "Maternal and child mortality reduction in recipient countries. "
        "Health system strengthening through official development assistance. "
        "WHO global health programs. KOICA health sector projects in developing countries."
    ),
    4: (
        "Education aid in developing countries through ODA. "
        "School construction and literacy programs in Africa and Asia. "
        "UNESCO education for all goals. Vocational training in recipient countries. "
        "Scholarship programs for developing country students. "
        "KOICA education projects in least developed countries."
    ),
    5: (
        "Gender equality and women empowerment in developing countries through ODA. "
        "Gender-based violence and child marriage in Africa Asia Middle East. "
        "Reproductive health programs in recipient countries. "
        "Girls education in developing countries. "
        "UN Women programs. Female empowerment through international development."
    ),
    6: (
        "Clean water and sanitation access in developing countries through ODA. "
        "WASH programs in Africa and Asia. "
        "Drinking water infrastructure aid in recipient countries. "
        "Wastewater treatment development assistance. "
        "Hygiene improvement projects in least developed countries."
    ),
    7: (
        "Energy access through ODA in developing countries. "
        "Renewable energy solar off-grid electricity in Africa and Asia. "
        "Clean cooking energy programs in recipient countries. "
        "Energy poverty reduction through international development assistance. "
        "KOICA energy projects in least developed countries."
    ),
    8: (
        "Economic growth and employment in developing countries through ODA. "
        "Job creation and labour rights in Africa and Asia. "
        "Trade capacity building in recipient countries. "
        "SME development assistance in least developed countries. "
        "Economic cooperation with developing countries through international aid."
    ),
    9: (
        "Infrastructure development ODA in developing countries. "
        "Road bridge port construction aid in Africa and Asia. "
        "Transport telecommunications industrialization in recipient countries. "
        "Technology transfer through international development assistance. "
        "KOICA infrastructure projects in developing countries."
    ),
    10: (
        "Inequality reduction in developing countries through ODA. "
        "Migration refugees remittances in international development context. "
        "Income gap and social exclusion in Africa and Asia. "
        "Inclusive development programs for marginalized populations in recipient countries."
    ),
    11: (
        "Urban development and disaster risk reduction in developing countries through ODA. "
        "Slum improvement housing programs in Africa and Asia. "
        "Disaster recovery assistance in recipient countries. "
        "Urban infrastructure development in least developed countries."
    ),
    12: (
        "Sustainable consumption and production in developing countries through ODA. "
        "Waste management and resource efficiency programs in Africa and Asia. "
        "Environmental sustainability in recipient countries through international aid. "
        "Green economy development in least developed countries."
    ),
    13: (
        "Climate change adaptation in developing countries through ODA. "
        "Flood drought disaster response in Africa Asia Pacific vulnerable countries. "
        "Climate finance Green Climate Fund for least developed countries. "
        "Sea level rise threat to small island developing states. "
        "Climate aid and adaptation programs in recipient countries."
    ),
    14: (
        "Marine and fisheries development in developing countries through ODA. "
        "Ocean conservation programs in Africa Asia Pacific. "
        "Sustainable fishing assistance in small island developing states. "
        "Coral reef protection and marine ecosystem international development aid."
    ),
    15: (
        "Forest and biodiversity conservation in developing countries through ODA. "
        "Desertification and land degradation response in Africa and Asia. "
        "Deforestation prevention international aid programs. "
        "Wildlife trafficking control and ecosystem restoration in recipient countries."
    ),
    16: (
        "Peace and governance support in developing countries through ODA. "
        "Post-conflict reconstruction in Africa Asia Middle East fragile states. "
        "Democracy rule of law anti-corruption programs in recipient countries. "
        "Election support human rights international development. "
        "KOICA governance projects in developing countries."
    ),
    17: (
        "ODA official development assistance Korea KOICA EDCF international cooperation. "
        "Development finance aid effectiveness in recipient countries. "
        "Capacity building technology transfer in developing countries. "
        "Korean foreign aid programs South Korea development cooperation. "
        "Partnership for sustainable development goals in recipient countries."
    ),
}


@dataclass
class SDGResult:
    sdg_label:     int           # top SDG 1-17, or 0 if below threshold
    sdg_score:     float         # similarity score of top SDG
    sdg_intensity: int           # 0-3 relevance level
    all_scores:    dict = field(default_factory=dict)  # {sdg: score} for all 17


@dataclass
class SDGMultiResult:
    """Multi-label result: all SDGs that score above SIM_THRESHOLD."""
    sdg_labels:    list          # all SDGs above threshold, sorted by score desc
    sdg_scores:    dict          # {sdg: score} for above-threshold SDGs only
    sdg_top:       int           # highest-scoring SDG (0 if none above threshold)
    sdg_top_score: float
    all_scores:    dict = field(default_factory=dict)


def _sim_to_intensity(score: float) -> int:
    if score < 0.35: return 0
    if score < 0.50: return 1
    if score < 0.65: return 2
    return 3


class SDGClassifier:
    """
    Two-step SDG classifier: Korean→English translation, then E5 cosine similarity.
    """

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._translator    = None   # Helsinki-NLP MarianMT pipeline
        self._embed_model   = None   # SentenceTransformer E5
        self._sdg_embeddings: np.ndarray | None = None
        self._sdg_ids: list[int] = list(range(1, 18))

    def _load_translator(self) -> None:
        if self._translator is not None:
            return
        from transformers import MarianMTModel, MarianTokenizer
        logger.info("Loading translation model %s on %s ...", TRANSLATE_MODEL, self.device)
        tokenizer = MarianTokenizer.from_pretrained(TRANSLATE_MODEL)
        model     = MarianMTModel.from_pretrained(TRANSLATE_MODEL)
        if self.device == "cuda":
            model = model.cuda()
        model.eval()
        self._translator = (tokenizer, model)
        logger.info("Translation model ready.")

    def _load(self) -> None:
        if self._embed_model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model %s on %s ...", EMBED_MODEL, self.device)
        self._embed_model = SentenceTransformer(EMBED_MODEL, device=self.device)
        anchors = [f"passage: {SDG_ANCHORS[sdg]}" for sdg in self._sdg_ids]
        self._sdg_embeddings = self._embed_model.encode(
            anchors,
            batch_size=17,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        logger.info("SDG embeddings ready. Shape: %s", self._sdg_embeddings.shape)

    def _translate_batch(self, texts: List[str], batch_size: int = 64) -> List[str]:
        self._load_translator()
        tokenizer, model = self._translator
        results: list[str] = []
        device = next(model.parameters()).device
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = [t[:400] for t in texts[i : i + batch_size]]
                tok = tokenizer(chunk, return_tensors="pt", padding=True,
                                truncation=True, max_length=256)
                tok = {k: v.to(device) for k, v in tok.items()}
                # Model defaults (num_beams=6, max_length=512) cost ~90x more
                # compute than needed for these short title/keyword strings and
                # don't improve output quality here (verified on real data) --
                # greedy decoding + a repetition guard is equally good and far
                # faster, since this feeds a similarity classifier, not human reading.
                gen = model.generate(**tok, num_beams=1, max_length=80, no_repeat_ngram_size=3)
                decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
                results.extend(decoded)
        return results

    def _fuse_scores(self, similarities: np.ndarray, original_ko: str) -> dict:
        """Apply keyword boost and return fused {sdg: score} dict."""
        kw    = keyword_scores(original_ko)
        fused = {sdg: float(similarities[i]) for i, sdg in enumerate(self._sdg_ids)}
        for sdg, hits in kw.items():
            boost = min(hits * KEYWORD_BOOST_PER_HIT, MAX_KEYWORD_BOOST)
            fused[sdg] = min(1.0, fused.get(sdg, 0.0) + boost)
        return fused

    def _fused_to_single(self, fused: dict) -> SDGResult:
        top   = max(fused, key=fused.__getitem__)
        score = fused[top]
        return SDGResult(
            sdg_label    = top if score >= SIM_THRESHOLD else 0,
            sdg_score    = round(score, 4),
            sdg_intensity= _sim_to_intensity(score),
            all_scores   = {k: round(v, 4) for k, v in fused.items()},
        )

    def _fused_to_multi(self, fused: dict) -> SDGMultiResult:
        above = {sdg: round(s, 4) for sdg, s in fused.items() if s >= SIM_THRESHOLD}
        ranked = sorted(above.keys(), key=above.__getitem__, reverse=True)
        top    = ranked[0] if ranked else 0
        return SDGMultiResult(
            sdg_labels    = ranked,
            sdg_scores    = above,
            sdg_top       = top,
            sdg_top_score = round(fused[top], 4) if top else 0.0,
            all_scores    = {k: round(v, 4) for k, v in fused.items()},
        )

    def _embed(self, translated: List[str], batch_size: int) -> np.ndarray:
        prefixed = [f"query: {t[:512]}" for t in translated]
        return self._embed_model.encode(
            prefixed, batch_size=batch_size,
            normalize_embeddings=True, show_progress_bar=False,
        )

    def classify_batch(self, texts: List[str], batch_size: int = 128) -> List[SDGResult]:
        """Single-label classification (backward compatible)."""
        self._load_translator()
        self._load()
        logger.info("Translating %d texts ko→en ...", len(texts))
        translated   = self._translate_batch(texts, batch_size=min(64, batch_size))
        similarities = self._embed(translated, batch_size) @ self._sdg_embeddings.T
        return [
            self._fused_to_single(self._fuse_scores(similarities[i], texts[i]))
            for i in range(len(texts))
        ]

    def classify_multilabel_batch(
        self, texts: List[str], batch_size: int = 128
    ) -> List[SDGMultiResult]:
        """Multi-label classification: returns all SDGs above SIM_THRESHOLD."""
        self._load_translator()
        self._load()
        logger.info("Translating %d texts ko→en (multi-label) ...", len(texts))
        translated   = self._translate_batch(texts, batch_size=min(64, batch_size))
        similarities = self._embed(translated, batch_size) @ self._sdg_embeddings.T
        return [
            self._fused_to_multi(self._fuse_scores(similarities[i], texts[i]))
            for i in range(len(texts))
        ]

    def classify(self, text: str) -> SDGResult:
        return self.classify_batch([text])[0]

    def classify_multilabel(self, text: str) -> SDGMultiResult:
        return self.classify_multilabel_batch([text])[0]
