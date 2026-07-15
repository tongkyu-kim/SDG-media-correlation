"""
Candidate rule for routing articles to the translate+embed BERT/E5 stage.

An article qualifies as a candidate if any of the following hold:
  - an explicit Korean ODA actor is mentioned (KOICA, EDCF, etc.)
  - it has at least one SDG keyword hit AND mentions an ODA-recipient country
  - it matches the broader development/organization/agenda vocabulary
    (see development_terms_ko.py)
  - a co-occurrence-only sector term (school, hospital, SME, etc.) appears
    together with any country mention

The country-mention signal is deliberately kept paired with a keyword-hit
requirement rather than treated as independent: populous ODA-recipient
countries (China, Vietnam, India, Indonesia, Thailand) are mentioned
constantly in Korean news for reasons unrelated to development, so an
unpaired country signal alone produces an unmanageably large candidate
population. Explicit ODA-actor mentions, the development-vocabulary match,
and co-occurring sector terms are specific enough to stand on their own.

Because Stage 3 (translation + anchor-based E5 similarity) is the precision
mechanism, false positives admitted here are inexpensive — they are filtered
out downstream. False negatives are not recoverable, since an excluded
article never reaches Stage 3. The rule is therefore tuned toward recall
within a bounded compute budget, rather than toward precision.

This module is the single implementation of the rule; run_classify.py,
precheck_classify.py, sample_for_labeling.py, and build_sampling_frame.py
all import it rather than reimplementing the logic inline.
"""

from __future__ import annotations

import pandas as pd

from reference.countries_ko import detect_countries, detect_oda_recipient_countries
from reference.development_terms_ko import has_development_vocab_hit, has_cooccur_only_sector_hit


def compute_signals(df: pd.DataFrame, kw_clf) -> dict[str, pd.Series]:
    """
    Compute all candidate signals for a raw article DataFrame. Returns a
    dict of boolean Series so callers can inspect individual signals (e.g.
    for diagnostics/stratum reporting), not just the final mask.
    """
    short = kw_clf._text_short(df)
    long_ = kw_clf._text_long(df, short)

    has_oda_country    = long_.apply(lambda t: bool(detect_oda_recipient_countries(t)) if t else False)
    has_any_country     = long_.apply(lambda t: bool(detect_countries(t)) if t else False)
    has_dev_vocab       = long_.apply(has_development_vocab_hit)
    has_cooccur_sector  = long_.apply(has_cooccur_only_sector_hit)

    return {
        "has_oda_country":   has_oda_country,
        "has_any_country":   has_any_country,
        "has_dev_vocab":     has_dev_vocab,
        # cooccur-only sector terms (학교/병원/중소기업/etc) only count when
        # a country is also mentioned — see development_terms_ko.py docstring.
        "has_cooccur_sector": has_cooccur_sector & has_any_country,
    }


def is_candidate(kw: pd.DataFrame, signals: dict[str, pd.Series], bert_min_hits: int = 1) -> pd.Series:
    """
    Candidate rule — an article is a candidate if ANY of:
      - policy_actor == 1                                   (explicit KOICA/EDCF/ODA actor mention — standalone)
      - kw_sdg_hits >= bert_min_hits  AND  ODA-recipient country mention  (paired, not independent — see module docstring)
      - matches the broader dev/org/agenda vocabulary (development_terms_ko.py) — standalone
      - a cooccur-only sector term together with any country mention — standalone pair
    """
    return (
        (kw["policy_actor"] == 1) |
        ((kw["kw_sdg_hits"] >= bert_min_hits) & signals["has_oda_country"]) |
        signals["has_dev_vocab"] |
        signals["has_cooccur_sector"]
    )


def classify_stratum(kw: pd.DataFrame, signals: dict[str, pd.Series], bert_min_hits: int = 1) -> pd.Series:
    """
    Three-way stratum label for annotation sampling / diagnostics:
      candidate   — sent to Stage 3 under the candidate rule (see is_candidate)
      borderline  — mentions SOME country (incl. non-ODA/donor countries)
                    but tripped no other signal; genuinely ambiguous
      negative    — no signal at all
    Borderline mainly captures donor-country mentions (USA/Japan/etc, which
    never count toward has_oda_country) and ODA-recipient-country mentions
    with zero keyword hits, which do not qualify as candidates on their own.
    """
    candidate = is_candidate(kw, signals, bert_min_hits)
    borderline = ~candidate & signals["has_any_country"]

    stratum = pd.Series("negative", index=kw.index)
    stratum[borderline] = "borderline"
    stratum[candidate] = "candidate"
    return stratum
