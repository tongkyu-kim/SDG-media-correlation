# SDG Classification Methodology

> This document records the rationale behind each methodological decision in the media classification pipeline, the empirical checks used to validate each choice, and known limitations. It is organized to anticipate the questions a peer reviewer would raise about measurement validity.

---

## 1. Research Design

### Unit of Analysis and Panel Structure

The study constructs three panel datasets:

| Panel | Unit | Observation | Key use |
|-------|------|-------------|---------|
| A | SDG × Month | Media attention to each SDG goal per month | Aggregate agenda-setting effect |
| B | Country × Month | Media attention to each ODA recipient country | Country-specific salience |
| **C** | **Country × SDG × Month** | **Media attention to recipient country–SDG pair** | **Primary identification** |

**Justification for Panel C as primary:** The research question is whether Korean media coverage of development issues in specific recipient countries predicts Korea's ODA allocation to those countries for those SDG sectors. Panel C provides the necessary bilateral variation — it can distinguish between, say, increased coverage of health issues in Ethiopia vs. Ethiopia generally, allowing for country and SDG fixed effects simultaneously.

Panel A is susceptible to global news cycles (e.g., a global health crisis inflates SDG3 for all countries simultaneously), which cannot be absorbed by country FE alone. Panel C is the theoretically correct level of aggregation for the donor–recipient–sector mechanism.

---

## 2. ODA Data: SDG Assignment

### 2.1 Three-Tier Assignment Priority

Each ODA project is assigned to a primary SDG using the following priority ladder, applied in order:

1. **Direct SDG tag** (22,542 projects, 27.1%): Projects with an explicit SDG marker in the KOICA/EDCF reporting data take priority. These are the most reliable assignments.

2. **CRS sector code crosswalk** (57,589 projects, 69.3%): Projects without a direct tag are assigned via the OECD CRS purpose code → SDG mapping from Pincet et al. (2019), *DCD/DAC/STAT(2015)9 Annex 3*. This is the standard academic instrument for SDG-ODA alignment and is used in OECD official reporting and multiple peer-reviewed studies on SDG financing.

3. **Policy markers** (0 projects after tiers 1–2 exhaust their coverage): Rio markers (environment, biodiversity, climate) and gender marker as tiebreaker — not needed in practice given the high coverage of tiers 1 and 2.

4. **Unassigned** (2,951 projects, 3.6%): Projects with no CRS sector code and no direct tag. These are excluded from SDG-level analysis but retained in total ODA counts.

**Validation:** 100% of the 83,082 projects in Korea's ODA dataset (2010–2023) were mapped to an ISO3 country code. The three-tier system covers 96.4% with an SDG label, which is high relative to comparable studies (OECD 2020 reports ~89% coverage for DAC members using similar methods).

### 2.2 Korea's Actual ODA Distribution (Sanity Check Anchor)

This distribution is critical for validating the media classifier outputs. If the media variable shows a very different distribution, that is a measurement validity concern, not necessarily a finding:

From `oda_sdg_annual.csv` (2010–2023 pooled disbursements):

| SDG | Description | Approx. share |
|-----|-------------|---------------|
| 9 | Industry, Infrastructure & Innovation | ~28–32% |
| 4 | Quality Education | ~20–25% |
| 3 | Good Health | ~10–15% |
| 6 | Clean Water & Sanitation | ~8–10% |
| 11 | Sustainable Cities | ~6–8% |
| 16 | Peace & Governance | ~5–7% |
| 2 | Zero Hunger | ~4–6% |
| *Others* | | remainder |

**Key implication:** SDG3 (Health) represents approximately 10–15% of Korea's ODA disbursements. A media classifier showing SDG3 at 46% of development-relevant articles is implausible as a measure of Korean public attention to *development* health issues, and was identified as a measurement artifact caused by domestic health content passing through the filter (see Section 3.3).

---

## 3. Media Classification Pipeline

The classification proceeds in three sequential stages, each with a specific role, applied to the full BigKinds news corpus (2007–2025, 228 monthly files).

### 3.1 Stage 1: Korean Keyword Pre-Filter

**What it does:** A rule-based classifier using Korean SDG keyword lists (derived from the SDSN keyword taxonomy and the Aurora SDG query system, translated and extended for Korean news vocabulary) assigns every article a candidate SDG label, an SDG hit count, and supplementary variables (aid_stance, issue_frame, crisis_type, policy_actor).

**Why it is necessary:**
- The corpus contains on the order of 15 million articles. Running a GPU-based embedding model on all articles is computationally prohibitive.
- Keyword pre-filtering is standard practice in large-scale computational text analysis (see Roberts et al. 2014; Grimmer & Stewart 2013). The pre-filter trades precision for recall at this stage: articles without any SDG-relevant keyword and no development-vocabulary or country signal are extremely unlikely to be development-relevant, so excluding them is low-risk, while an article admitted at this stage is refined for precision downstream (Stage 3).
- The keyword classifier produces auxiliary variables (aid_stance, issue_frame, etc.) that serve as independent measures and robustness checks.

**Keyword list curation:** Two classes of terms were identified as sources of false positives and removed from the SDG keyword lists (`keywords_ko.py`):

1. **Homonym collisions.** 의원 was listed under SDG3 to mean "clinic," but the identical string also means "lawmaker/assemblyman" (as in 국회의원), which fired SDG-3 hits on ordinary political news.
2. **Generic domestic vocabulary.** Terms such as 경제성장, 최저임금, 비정규직 (SDG8), 4차 산업혁명, 디지털 전환 (SDG9), 불평등, 소득격차 (SDG10), and ESG (SDG12) appear routinely in ordinary Korean domestic economic and political reporting with no connection to a developing country's situation, inflating keyword-hit counts on domestic articles.

**Validation:**
- Agreement between keyword SDG label and ML classifier label: **87–89%** (pre-check on a 2016 sample of 1,060 candidates). The 11–13% disagreement represents cases where the ML stage provides a better label — expected behavior of a hybrid system in which ML refines rather than replaces rule-based output.
- See Section 4.4 for candidate-stratum precision and recall estimates from hand-coded validation data.

**Limitation to disclose:** Keyword lists have language-specific coverage gaps. Terms that appear as English-language loanwords in Korean news (e.g., "ODA," "SDG") are captured; idiomatic Korean expressions for the same concept without a canonical keyword may be missed. The ML stage is designed to recover some of these cases via semantic similarity.

### 3.2 Stage 2: Development Signal Filter

**What it does:** Articles are routed to the ML stage if any of the following hold: (a) an explicit Korean ODA actor is mentioned (KOICA, EDCF, 공적개발원조); (b) the article has at least one SDG keyword hit **and** mentions a country from Korea's ODA recipient set; (c) the article matches a broader development/organization/agenda vocabulary (international organizations, Korean development actors beyond KOICA/EDCF, development/SDG-agenda/humanitarian/climate/trade-infrastructure vocabulary, and major development initiatives — `development_terms_ko.py`); or (d) a co-occurrence-only sector term (school, hospital, SME, etc.) appears together with any country mention.

**Theoretical justification:** The mechanism under test is *Korean domestic media coverage of development issues in specific recipient countries → Korean government ODA allocation decisions for those countries*. Media coverage must be *about* a recipient country to be relevant to the allocation decision — an article about Korean domestic health policy, even using development-relevant health vocabulary, does not belong to the treatment variable. This filtering step operationalizes "development-relevant media coverage," analogous to geographic targeting in other media-ODA studies (Eisensee & Strömberg 2007 use disaster-country mentions as the unit of relevance; Balcells et al. use country-specific conflict coverage).

**Design choice: ODA recipient countries only (not all countries).** The country signal is restricted to Korea's ODA recipient set rather than all countries, for two reasons:

1. *Construct validity:* A Korean health article mentioning the US FDA or German pharmaceutical research does not constitute Korean public attention to *development* health issues. Including donor-country mentions introduces systematic upward bias in SDG3 (health, via FDA/US pharma mentions) and SDG8 (labour, via US/EU economic news).
2. *Empirical evidence:* with all-country detection, candidates ran to roughly 6% of a test file and SDG3 accounted for 46% of classified articles — far exceeding Korea's actual ~12% SDG3 share of ODA. Restricting to ODA-recipient countries cut candidates by roughly 46%, driven almost entirely by removing developed-country mentions.

**Technical implementation:** Country names in Korean are detected via a compiled lookup table of roughly 190 countries with Korean-language names sourced from the UN Term Portal and Korean MOFA standards. Short Korean country names (≤3 characters: 이란=Iran, 수단=Sudan, 오만=Oman, 피지=Fiji) use a negative lookbehind regex `(?<![가-힣])` to prevent false matches against common Korean syllables (e.g., "건강이란 무엇인가," "what is health called," should not trigger Iran detection). The ODA-recipient country set is loaded at runtime from `oda_country_sdg_annual.csv`, ensuring consistency between the ODA dataset and the media filter; if that file is missing, the pipeline logs a warning and falls back to all-country detection rather than degrading silently.

**Design rationale — recall-oriented signal combination.** An earlier, stricter version of this rule required a co-occurring SDG keyword hit (≥2) for every country-mention signal. Cross-tabulating hand-coded validation data by stratum showed this conjunction was the specific mechanism behind the filter's recall loss: cases with only one of the two required signals were excluded before ever reaching the ML stage. Since the ML stage (Stage 3, translation + anchor-based similarity) is the actual precision mechanism, the filter was redesigned to prioritize recall: the keyword-hit threshold for the country-paired signal was lowered, and additional signals (explicit ODA-actor mentions, the development vocabulary dictionary, and co-occurring sector terms) were added as independent triggers.

The country-mention signal is deliberately *not* treated as fully independent, however: a calibration test showed that admitting any ODA-recipient-country mention on its own (with no keyword requirement) raises the candidate rate roughly 20-fold, driven almost entirely by populous ODA recipients (China, Vietnam, India, Indonesia, Thailand) that appear in routine Korean news for reasons unrelated to development. The chosen design keeps the country signal paired with a keyword-hit requirement specifically to avoid this, while treating the more specific signals (ODA-actor mentions, development vocabulary, sector-term co-occurrence) as independent triggers. See Section 4.6 for the calibration figures.

**Limitation to disclose:** Articles about global multilateral development initiatives (e.g., G20 development finance communiqués, UNGA SDG debates) that do not name specific recipient countries are excluded from the media variable, introducing a downward bias for SDG17 (Partnerships/Financing for Development), which is often discussed in aggregate terms. SDG17 counts should be interpreted as a lower bound.

Separately, China remains a technical ODA recipient in Korea's data (historical KOICA/EDCF loans) despite functioning as a major power in most Korean news coverage — most Korean articles mentioning China are not about "a developing country's development situation" in the sense this filter intends. This is not corrected in the country filter itself (doing so would require excluding a country that is a genuine, if atypical, recipient in the underlying ODA data); it is handled at the human-coding stage via the annotation codebook (Section 4.4) but remains unresolved in the deployed classifier — an open item.

### 3.3 Stage 3: Korean → English Translation + Multilingual Sentence Embedding

**What it does:** Articles passing Stage 2 are (a) translated from Korean to English via Helsinki-NLP/opus-mt-ko-en, then (b) embedded with intfloat/multilingual-e5-base (Wang et al. 2024), then (c) compared against English-language ODA/development-specific SDG anchor texts via cosine similarity.

**Why translation is necessary:**

Multilingual embedding models like multilingual-e5-base map texts from all supported languages into a shared semantic space. In that space, the Korean word for "health" and the English phrase "global health aid" are nearby regardless of whether the Korean article is about a domestic clinic or a KOICA health project in Ethiopia. The embedding model cannot represent the distinction *domestic Korean context vs. international development context* on its own — this distinction is a domain-specificity feature, not a language feature.

Translation to English decouples the language from the domain. After translation:
- "발기약은 혈관 협심증약 함께 먹으면 위험" → "Taking erectile dysfunction medication with heart medication is dangerous" → low cosine similarity with "International health aid in developing countries; malaria HIV AIDS response in Africa"
- "에티오피아 모성사망률 감소를 위한 KOICA 보건사업" → "KOICA health project for reducing maternal mortality in Ethiopia" → high cosine similarity with same anchor

**Model choice:**

| Component | Model | Rationale |
|-----------|-------|-----------|
| Translation | `Helsinki-NLP/opus-mt-ko-en` (MarianMT, 74M params) | Established Korean→English MT, runs locally, no API cost/dependency, adequate quality for short-form news text |
| Embedding | `intfloat/multilingual-e5-base` (278M params) | Wang et al. (2024), state-of-the-art multilingual embeddings, natively supports Korean and English, trained with instruction-following prefix ("query:") |
| Anchors | English-only, ODA-development-framed | 17 anchor texts, each explicitly anchored to international development, recipient countries, ODA language; no domestic Korean context |

**Rejected alternatives:**

- *Translation + OSDG text classifier:* OSDG (osdg.ai) is trained specifically on UN/OECD development documents, but on scientific publications, and may over-classify academic abstracts unrelated to developing-country media. It also assigns labels based on absolute scores rather than relative semantic similarity, which may reject short translated news titles as below threshold. Cosine similarity is more robust to short-text inputs.
- *Multilingual-e5-base without translation:* Korean embedding space conflates domestic and development health/education contexts, producing SDG3 = 46% vs. Korea's actual ODA share of ~12% — a 3–4× overestimate that would bias the media→ODA coefficient upward for SDG3.
- *Full BERT classification on the entire corpus without staged filtering:* infeasible at the corpus's scale on available hardware; the staged pipeline (keyword → development signal → ML) is what makes the ML step tractable.

**Anchor text design:** SDG anchors are single-label prototypes. Each anchor explicitly contains (1) the phrase "in developing countries," "in recipient countries," or "through ODA"; (2) relevant geographic identifiers (Africa, Asia, least developed countries); (3) institutional references (KOICA, EDCF, WHO, WFP, UN agencies); (4) English translations of key Korean ODA vocabulary. This ensures the embedding comparison is sensitive to the development-context dimension, not just topic — a Korean article about domestic climate legislation translates to text without "developing countries," "ODA," or "Africa," and scores low against all 17 SDG anchors.

**Threshold calibration:** `SIM_THRESHOLD = 0.35` for the English-input configuration — lower than an initial multilingual-input configuration (0.45), because English–English cosine similarity in the E5 space distributes differently from Korean–English similarity. The threshold accepts clearly development-relevant translations while rejecting domestic-context ones. A keyword boost (+0.03 per domain-specific Korean keyword hit, capped at +0.15) is applied to the cosine score as a soft correction for articles where high-quality Korean development keywords (e.g., "공적개발원조," "KOICA") appear despite imperfect translation.

**Translation decoding parameters:** `num_beams=1`, `max_length=80`, `no_repeat_ngram_size=3` are used rather than the MarianMT checkpoint's defaults (`num_beams=6`, `max_length=512`). On GPU hardware (Tesla T4) the default configuration measured ~1 article/sec; the revised configuration measures ~26 articles/sec with no observed quality difference on inspected samples, since the translation output feeds a similarity classifier rather than being read directly. The 6-beam/512-token defaults were not improving translation quality on this input (titles and comma-separated keyword fragments, not full sentences) — both configurations produced the same degenerate repetition artifacts on SEO-padded keyword fields (a keyword-stuffed source list translating to the same term repeated many times), which largely reflects the source data rather than a decoding failure. `max_length=80` is sized for the actual output length of translated titles/keyword fragments.

---

## 4. Validation Strategy

### 4.1 Pre-check Protocol

Before running the full corpus, the pipeline is validated on a held-out file and inspected for:

1. **Candidate reduction check:** whether the staged filter reduces articles to a plausible development-relevant subset.
2. **SDG distribution check:** whether the media SDG distribution is broadly consistent with Korea's ODA SDG distribution at the aggregate level. Perfect correlation is not expected (media leads ODA by design), but extreme divergence (e.g., SDG3 far above its ODA share) is a red flag.
3. **Manual spot-check:** top-3 articles per SDG by embedding similarity score, reviewed manually for development relevance.
4. **Keyword–ML agreement rate:** target >80%. Higher agreement indicates the keyword and ML classifiers identify the same underlying construct; systematic disagreement would signal a classification inconsistency.

### 4.2 Construct Validity: Expected Correlations

The media variable (`media_sdg_count_ct`) for Panel C should exhibit:

- **Positive serial correlation within country–SDG pairs** (development issues are covered in clusters).
- **Cross-country co-movement during global events** (e.g., a regional famine spikes SDG2 for affected countries; an epidemic spikes SDG3 for affected regions) — this is signal, not noise.
- **Near-zero correlation between media coverage of Korea's domestic health policy and SDG3 ODA disbursements.** A large positive correlation here would suggest residual domestic contamination and should prompt robustness checks using the country-restricted sample.

### 4.3 Robustness Checks to Report

1. **Keyword-only specification:** replace the translated-embedding media variable with the keyword-only classifier. Results should be directionally consistent but attenuated.
2. **High-confidence-only subsample:** restrict to articles at the highest similarity-score intensity tier. This should sharpen coefficients if the measurement is valid.
3. **Policy-actor articles only:** restrict to articles with an explicit KOICA/EDCF mention — a high-precision, low-recall subset; coefficients should be larger in magnitude.
4. **ODA recipient country subset:** restrict to countries receiving more than a threshold amount per year, to test whether the main result is driven by major bilateral partners.
5. **Lag structure sensitivity:** test media lags of 3, 6, 9, 12, and 18 months. ODA programming cycles typically run 12–24 months; media effects at very short lags (1–2 months) are implausible given procurement timelines and should be near zero.

### 4.4 Human Annotation Sample Design

**Problem addressed:** a plain simple-random sample of Korean news is overwhelmingly non-development content, since only a small minority of articles pass even a loose relevance filter. Coding such a sample leaves coders with too few positive or boundary cases to calibrate against each other, which depresses inter-coder agreement independent of the classifier's actual quality.

**Design:** `sample_for_labeling.py` pre-scores every article with the keyword classifier and country/vocabulary detectors before sampling, using the same criteria the classification pipeline uses to route articles to the ML stage:

- **candidate** — passes the Stage 1+2 development-signal filter (Section 3.2). Mirrors the population actually scored by the ML stage.
- **borderline** — some SDG keyword hits or a country mention, but not enough to qualify as a candidate. The genuinely ambiguous cases that build coder calibration.
- **negative** — no keyword hits, no country mention. A smaller control group confirming the true-negative rate.

Each year's sampling quota is drawn proportionally across strata (default 50% candidate / 30% borderline / 20% negative) rather than uniformly at random, oversampling the population the classifier is actually scored against.

**Reporting caveat:** because this is an enriched, not simple-random, sample, the prevalence of positive labels within it does **not** estimate corpus-wide prevalence and should not be reported as such. Precision/recall/F1 computed within the `candidate` stratum are valid estimates for that stratum specifically — which is also the exact population the trained classifiers are applied to, making it the appropriate stratum for primary validation metrics.

**Sample and inter-coder reliability:** 595 articles were hand-coded by two independent coders (seed-fixed enriched stratified draw; stratum counts candidate=277/borderline=164/negative=119 after excluding rows with unresolved disagreement), with a 150-row overlap block coded by both.

| Metric | Value | Scope |
|---|---|---|
| Cohen's κ | 0.885 | 150 overlap rows |
| Krippendorff's α | 0.885 | 150 overlap rows |
| Krippendorff's α | 0.739 | 445 non-overlap rows (both coders independently labeled the full sample) |
| **Krippendorff's α** | **0.772** | **all 595 rows** |

Reliability improved over the course of coding through one codebook refinement. An initial pass produced κ=0.577 (below the conventional 0.60 acceptability threshold), driven by a specific, resolvable disagreement: one coder correctly extended "development-relevant" to cover conflict/crisis stories in developing countries (coups, earthquakes, epidemics) that do not mention Korean aid, but over-applied the same extension to domestic-Korean and developed-country stories that merely touched an SDG-adjacent theme. Applying a single mechanical rule — evaluate whether the story's main geographic subject is a developing/ODA-recipient country before any thematic judgment — resolved the majority of disagreements and produced the reliability figures above. This full "does this discuss a developing country's development situation" rule (poverty, health, education, conflict, climate, aid) is documented as the coding standard; a fresh, blind (no coder discussion) validation round is the natural next check on whether this reliability generalizes.

Both coders independently hand-coded the full 595 rows rather than only the designated overlap, which is why Krippendorff's α (which tolerates partial/unbalanced coverage) is reported alongside Cohen's κ (which requires complete paired coverage) — α on the full 595 is the more representative reliability estimate.

**Candidate-stratum precision and recall:** cross-tabulating the 595-row sample by stratum against `label_development_relevant`:

| Stratum | n | positives |
|---|---|---|
| candidate | 277 | 69 |
| borderline | 164 | 4 |
| negative | 119 | 0 |

Recall = 69/73 = **94.5%** (Wilson 95% CI: 86.7–97.8%); candidate-stratum precision = 69/277 = **24.9%** (Wilson 95% CI: 20.2–30.3%). Both intervals are wide because the sample contains only 73 positives in total — the practical constraint motivating larger follow-up annotation rounds. The recall-leak cases (4 of 73 positives, in the borderline stratum) motivated the Stage 1+2 filter redesign described in Section 3.2.

**Implementation issues identified and corrected during development, disclosed for transparency:**

1. The ODA-recipient country crosswalk file the filter depends on (`oda_country_sdg_annual.csv`) was, at one point, missing from its expected path; when missing, the filter silently falls back to matching any of the ~190 countries in the lookup table rather than only ODA recipients, which inflates false positives from donor-country mentions. A logged warning now makes this failure mode visible rather than silent.
2. The homonym and generic-vocabulary keyword issues described in Section 3.1.

Combined, correcting both issues raised candidate-stratum precision on the hand-coded validation sample from roughly 9–10% to the 24.9% reported above.

### 4.5 Supervised Classifier Training Results

Both supervised classifiers were trained on the 560 hand-coded rows where both coders agreed on `label_development_relevant` (35 of 595 rows had unresolved disagreement and were excluded from training data rather than tie-broken, to avoid injecting label noise; 487 negative / 73 positive).

**`train_oda_classifier.py`** (TF-IDF char n-gram + LogisticRegression, 5-fold stratified CV):

| Metric | Value |
|---|---|
| Accuracy | 0.909 ± 0.014 |
| F1 | 0.553 ± 0.083 |
| Precision | 0.766 ± 0.101 |
| Recall | 0.437 ± 0.077 |
| ROC-AUC | 0.945 ± 0.020 |

High ROC-AUC with modest recall at the default 0.5 threshold indicates the model ranks positives well but is conservative given the small positive class (73 examples); recall could likely be improved by lowering the decision threshold, at a precision cost, if a downstream use case prioritizes recall.

**`train_devrel_classifier.py`** (fine-tuned `klue/roberta-base`, 4 epochs, 20% held-out validation, CPU):

| Metric | Value (best epoch) |
|---|---|
| F1 | 0.667 |
| Accuracy | 0.893 |
| ROC-AUC | 0.946 |

**Note on the `oda_relevant` vs. `label_development_relevant` distinction:** `train_oda_classifier.py` was designed around a narrower `oda_relevant` concept (its seed keyword list is ODA/aid-program-specific: KOICA, EDCF, 공적개발원조), while the annotation codebook coders used asks the broader "does this discuss a developing country's development situation" question (`label_development_relevant`). For this training run, `label_development_relevant` was used directly as `oda_relevant` rather than conducting a separate narrower labeling round — a simplification worth flagging, since the two constructs are related but not identical, and the reported metrics reflect the broader construct.

Both the small positive class (73 examples) and the limited total sample size (595 hand-coded rows, given the low base rate of development-relevant content) are treated as an open constraint rather than a settled question: additional annotation rounds are planned specifically to enlarge the positive-class sample and tighten the precision/recall intervals above.

### 4.6 Candidate-Rule Calibration

The candidate rule described in Section 3.2 was chosen after comparing two specifications on a representative monthly file (n=68,428 articles):

| Specification | Candidate rate | Full-corpus (2007–2025) extrapolation |
|---|---|---|
| Country-paired keyword requirement (chosen) | 12.9% | ~1.8M articles, ~20 GPU-hours |
| Country mention treated as fully independent | 33.7% | ~4.9M articles, ~53 GPU-hours |

The fully independent specification was rejected: it is driven almost entirely by populous ODA-recipient countries (China, Vietnam, India, Indonesia, Thailand) appearing in routine Korean news for reasons unrelated to development, and the resulting compute cost is disproportionate to the additional recall it would provide over the chosen specification. For comparison, the earlier, stricter AND-conjunction rule (Section 3.2) measured a 1.7% candidate rate on the same file — the chosen specification increases the candidate population roughly 7-fold relative to that baseline while remaining a small fraction of the full corpus.

---

## 5. Anticipated Peer Review Challenges and Responses

### Challenge 1: "Why Korean news? Korean public doesn't determine ODA allocations."

**Response:** Korea's ODA governance is domestically contested. KOICA and EDCF allocations are subject to parliamentary appropriations (ODA Act 2010; amended 2020) and the Prime Minister's ODA Policy Committee includes representatives from civil society and academia, both of whom are influenced by media discourse. Moreover, Korea's ODA has been shown to be responsive to domestic political economy factors (Park 2017; Kim & Kim 2013). The domestic audience model of foreign aid (Milner & Tingley 2013) does not require public opinion to directly determine allocations — media salience shapes bureaucratic and parliamentary attention, which affects allocation through multiple indirect channels.

### Challenge 2: "Your media variable is contaminated by domestic Korean news."

**Response:** This is precisely the concern that motivated the staged filtering design. The development-signal filter (Stage 2) restricts articles to those mentioning ODA-recipient countries or explicit development vocabulary, removing articles that discuss domestic Korean SDG issues without international development relevance. The translation + development-anchored embedding (Stage 3) further discriminates by requiring the translated article's semantic content to align with ODA/development-specific anchor texts, not just the SDG topic in general. Residual contamination is acknowledged as a limitation but substantially reduced, and the robustness check using `policy_actor = 1` articles provides a high-precision upper bound.

### Challenge 3: "Machine translation introduces noise. Why not use a Korean-language classifier trained on Korean development text?"

**Response:** No Korean-language training dataset for ODA/development SDG classification exists to our knowledge. OSDG and Aurora, the two established SDG text classification systems, were trained on English-language documents. Using Helsinki-NLP translation as a preprocessing step allows this pipeline to leverage these English-calibrated semantic spaces without requiring Korean-labeled training data. The keyword boost applied to the original Korean text provides a correction signal when key Korean development terms appear, reducing dependence on translation quality for the most important articles. As a robustness check, the keyword-only specification does not require translation and serves as an alternative identification strategy.

### Challenge 4: "How do you handle articles relevant to multiple SDGs?"

**Response:** The classifier assigns a primary SDG label (highest cosine similarity + keyword boost) and a secondary SDG label (second highest, if cosine similarity ≥ 0.20). The main analysis uses primary SDG only; robustness checks with secondary-SDG-inclusive counting are reported. This follows the approach of Sacchi et al. (2019) and the OSDG documentation, which recommend single-label classification for index construction to avoid double-counting. Articles with very close primary/secondary scores are flagged for multi-SDG treatment in sensitivity analysis.

### Challenge 5: "Your ODA SDG assignment via CRS crosswalk is imprecise."

**Response:** The Pincet et al. (2019) CRS–SDG crosswalk is the OECD's own recommended instrument for retrospective SDG tagging of ODA flows and has been used in peer-reviewed studies (Bhattacharya et al. 2018; Ohno et al. 2020; OECD DAC 2020 SDG Finance report). The alternative — project-level text classification of Korean ODA project descriptions — would require a Korean-language ODA text classifier that does not currently exist and introduces its own measurement uncertainty. The crosswalk approach is conservative and consistent with how comparable studies operationalize ODA-SDG alignment. Direct SDG tags (27.1% of projects) are used when available and take priority.

---

## 6. Data Disclosure and Replication Notes

| Item | Status |
|------|--------|
| BigKinds news data | Licensed dataset; raw files not public but available to researchers from BigKinds upon institutional request |
| ODA data source | Korea KOICA/EDCF official data; available from OECD CRS database and KOICA Open Data Portal |
| CRS–SDG crosswalk | Pincet et al. (2019), publicly available from OECD iLibrary |
| Translation model | `Helsinki-NLP/opus-mt-ko-en`, publicly available on HuggingFace (MIT License) |
| Embedding model | `intfloat/multilingual-e5-base`, publicly available on HuggingFace (MIT License) |
| Pipeline code | All preprocessing, classification, and panel construction code available in this repository |
| Keyword lists | In `pipeline/classify/keywords_ko.py`; reviewable and extensible |
| Development vocabulary | In `pipeline/reference/development_terms_ko.py`; reviewable and extensible |
| Country–ISO3 mapping | In `pipeline/reference/countries_ko.py`; based on UN Term Portal and MOFA Korea standards |
| Human annotation sample | Enriched stratified design (candidate/borderline/negative), see Section 4.4; generated by `pipeline/sample_for_labeling.py` |
