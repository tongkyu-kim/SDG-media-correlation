# SDG Classification Methodology: Internal Notes for Peer Review Preparation

> **Purpose:** This document records the rationale behind each methodological decision in the media classification pipeline. It is written for internal use but structured as a response to anticipated peer review challenges. All choices are documented with (a) the theoretical justification, (b) the empirical check used to validate the choice, and (c) known limitations and how they are disclosed.

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

The classification proceeds in three sequential stages, each with a specific role. All stages are applied to each weekly BigKinds news file (697 files, 2010–2023).

### 3.1 Stage 1: Korean Keyword Pre-Filter

**What it does:** A rule-based classifier using Korean SDG keyword lists (derived from SDSN keyword taxonomy and Aurora SDG query system, translated and extended for Korean news vocabulary) assigns every article a candidate SDG label, an SDG hit count, and supplementary variables (aid_stance, issue_frame, crisis_type, policy_actor).

**Why it is necessary:**
- The corpus contains approximately 18 million articles. Running a GPU-based embedding model on all articles is computationally prohibitive (~12,000 hours at naive rates).
- Keyword pre-filtering to ~26% of articles is standard practice in large-scale computational text analysis (see Roberts et al. 2014; Grimmer & Stewart 2013). The pre-filter trades recall for precision: articles without any SDG-relevant keyword are extremely unlikely to be development-relevant.
- The keyword classifier produces auxiliary variables (aid_stance, issue_frame, etc.) that serve as independent measures and robustness checks.

**Validation:**  
- Agreement between keyword SDG label and ML classifier label: **87–89%** (from pre-check on 2016 sample of 1,060 candidates)
- The 11–13% disagreement represents cases where ML provides a better label — this is the expected behaviour of a hybrid system where ML refines rather than replaces rule-based output.

**Limitation to disclose:** Keyword lists have language-specific coverage gaps. Terms that appear in English-language loanwords in Korean news (e.g., "ODA," "SDG") are captured; idiomatic expressions for the same concept without the canonical keyword may be missed. The ML stage is specifically designed to recover some of these cases via semantic similarity.

### 3.2 Stage 2: ODA Recipient Country Filter

**What it does:** From keyword-flagged articles, only those that mention at least one country from the set of Korea's 147 ODA recipient countries (or have an explicit Korean ODA actor mention: KOICA, EDCF, 공적개발원조) are sent to the ML classifier.

**Theoretical justification:**  
The media effects mechanism we are testing is: *Korean domestic media coverage of development issues in specific recipient countries → Korean government ODA allocation decisions for those countries.* This implies that media coverage must be *about* a recipient country to be relevant to the allocation decision. An article about Korean domestic health policy, even if it uses development-relevant health vocabulary, does not belong to the treatment variable.

This filtering step operationalizes the concept of "development-relevant media coverage." This is analogous to geographic targeting in media-ODA studies: Eisensee & Strömberg (2007) use disaster country mentions as the unit of relevance; Balcells et al. use country-specific conflict coverage. Country mention is a necessary condition for bilateral development media relevance.

**Design choice: ODA recipient countries only (not all countries)**  
The filter was restricted to countries appearing in Korea's ODA recipient data rather than all countries, for two reasons:

1. *Construct validity:* If a Korean health article mentions the United States FDA or German pharmaceutical research, this does not constitute Korean public attention to *development* health issues. Including OECD donor country mentions would introduce systematic upward bias in SDG3 (health, due to FDA/US pharma mentions) and SDG8 (labour, due to US/EU economic news).

2. *Empirical evidence:* Pre-check analysis confirmed this bias. With all-country detection, 1,057 articles passed the filter (6.0%) and SDG3 accounted for 46% of classified articles — far exceeding Korea's actual ~12% SDG3 share of ODA. With ODA-recipient-country-only detection, candidates dropped to 569 (3.3%), a 46% reduction driven almost entirely by removing developed-country mentions.

**Technical implementation:**  
- Country names in Korean are detected using a compiled lookup table of ~170 countries with Korean-language names from UN Term Portal and Korean MOFA standards.
- Short Korean country names (≤3 characters: 이란=Iran, 수단=Sudan, 오만=Oman, 피지=Fiji) use negative lookbehind regex `(?<![가-힣])` to prevent false matches against common Korean syllables (e.g., "건강이란 무엇인가" — "what is health called" — should not trigger Iran detection).
- The recipient country set is loaded at runtime from `oda_country_sdg_annual.csv`, ensuring consistency between the ODA dataset and the media filter.

**Limitation to disclose:**  
Articles about global multilateral development initiatives (e.g., G20 development finance communiqués, UNGA SDG debates) that do not name specific recipient countries are excluded from the media variable. This introduces a downward bias for SDG17 (Partnerships/Financing for Development), which is often discussed in aggregate terms. This is acknowledged and SDG17 counts should be interpreted as a lower bound.

### 3.3 Stage 3: Korean → English Translation + Multilingual Sentence Embedding

**What it does:** Articles passing Stage 2 are (a) translated from Korean to English via Helsinki-NLP/opus-mt-ko-en, then (b) embedded with intfloat/multilingual-e5-base (Wang et al. 2024), then (c) compared against English-language ODA/development-specific SDG anchor texts via cosine similarity.

**Why translation is necessary:**

Multilingual embedding models like multilingual-e5-base map texts from all supported languages into a shared semantic space. In that space, the Korean word for "health" and the English phrase "global health aid" are nearby regardless of whether the Korean article is about a domestic clinic or a KOICA health project in Ethiopia. The embedding model is trained on broad multilingual corpora and cannot represent the distinction *domestic Korean context vs. international development context* — this distinction is not a language feature, it is a domain-specificity feature.

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

- *Translation + OSDG text classifier:* OSDG (osdg.ai) is trained specifically on UN/OECD development documents. However, the open OSDG model was trained on scientific publications and may over-classify academic abstracts unrelated to developing-country media. More critically, OSDG assigns labels based on absolute scores (trained distribution), not relative semantic similarity — it may reject all short translated news titles as below threshold. The cosine similarity approach is more robust to short-text inputs.

- *Multilingual-e5-base without translation (original approach):* As documented above, Korean embedding space conflates domestic and development health/education contexts. This produced SDG3 = 46% vs. Korea's actual ODA share of ~12%, a 3–4× overestimate that would bias the media→ODA coefficient upward for SDG3.

- *Helsinki-NLP translation + full BERT classification on all 18M articles:* Runtime ~12,000 hours on available hardware. Not feasible; the staged pipeline (keyword → country → ML) reduces the ML step to ~3% of the corpus.

**Anchor text design:**

SDG anchors are single-label prototypes. Each anchor explicitly contains:
1. The phrase "in developing countries" or "in recipient countries" or "through ODA"
2. Relevant geographic identifiers (Africa, Asia, least developed countries)
3. Institutional references (KOICA, EDCF, WHO, WFP, UN agencies)
4. English translations of key Korean ODA vocabulary

This design ensures the embedding comparison is sensitive to the development context dimension, not just the topic dimension. A Korean news article about domestic climate legislation translates to text without "developing countries," "ODA," or "Africa" — and will therefore score low against all 17 SDG anchors.

**Threshold calibration:**

- `SIM_THRESHOLD = 0.35` for the English-input configuration
- Note: this threshold is *not* the same as the earlier multilingual configuration (0.45), because English–English cosine similarity in the E5 space distributes differently than Korean–English similarity. The threshold was set to accept clearly development-relevant translations while rejecting domestic-context translations.
- Additionally: keyword boost (+0.03 per domain-specific Korean keyword hit, capped at +0.15) is applied to the cosine score. This acts as a soft smoothing for articles where high-quality Korean development keywords (e.g., "공적개발원조," "KOICA") appear despite imperfect translation.

---

## 4. Validation Strategy

### 4.1 Pre-check Protocol

Before running the full 697-file corpus (~18M articles), the pipeline was validated on a held-out file (`NewsResult_20160111-20160117.csv`, n=17,592 articles) and inspected:

1. **Candidate reduction check:** Whether the staged filter reduces articles to a plausible development-relevant subset (target: < 5% of corpus, > 50% identifiably development-relevant on spot-check).

2. **SDG distribution check:** Whether the media SDG distribution is broadly consistent with Korea's ODA SDG distribution at the aggregate level. Perfect correlation is not expected (media leads ODA by design), but extreme divergence (e.g., SDG3 = 46% in media vs. 12% in ODA) is a red flag.

3. **Manual spot-check:** Top-3 articles per SDG by E5 score are reviewed manually for development relevance. At least 2 of 3 should be identifiably development-relevant (not domestic Korean content).

4. **Keyword–ML agreement rate:** Target > 80%. Higher agreement indicates the keyword classifier and ML classifier are identifying the same underlying construct; systematic disagreement would signal a classification inconsistency.

### 4.2 Construct Validity: Expected Correlations

The media variable (`media_sdg_count_ct`) for Panel C should exhibit:

- **Positive serial correlation within country–SDG pairs** (development issues are covered in clusters)
- **Cross-country co-movement during global events** (Syria famine years spike SDG2 for MENA countries; Ebola years spike SDG3 for West Africa) — this is signal, not noise
- **Near-zero correlation between media coverage of Korea's domestic health policy and SDG3 ODA disbursements** — if this correlation is large and positive after the revised classifier, it suggests residual domestic contamination and should prompt robustness checks using the Panel C country-restricted sample

### 4.3 Robustness Checks to Propose in Paper

1. **Keyword-only specification:** Replace the translated-E5 media variable with the keyword-only classifier. Results should be directionally consistent but attenuated (less precise labeling).
2. **High-confidence-only subsample:** Restrict to articles with SDG intensity = 3 (highest E5 similarity). This should sharpen coefficients if the measurement is valid.
3. **Policy actor articles only:** Restrict to articles with `policy_actor = 1` (explicit KOICA/EDCF mention). This is a very high-precision but low-recall subset — coefficients should be larger in magnitude.
4. **ODA recipient country subset:** Restrict Panel C to countries receiving >$1M/year (removes noise from small/intermittent recipients). This tests whether the main result is driven by major bilateral partners.
5. **Lag structure sensitivity:** Test media lags of 3, 6, 9, 12, and 18 months. ODA programming cycles are typically 12–24 months; media effects at very short lags (1–2 months) are implausible given procurement timelines and should be near zero.

### 4.4 Human Annotation Sample Design

**Problem identified:** The initial annotation sample (n=638/coder) was drawn by simple random sampling stratified only by year. Since only ~3–6% of raw Korean news articles pass even the loose keyword+country relevance filter (Sections 3.1–3.2), a plain random sample is overwhelmingly non-development content. Coding that skewed sample left coders with too few positive/boundary cases to calibrate against each other, producing poor inter-coder agreement. That sample was abandoned before completion.

**Revised design:** `sample_for_labeling.py` now pre-scores every candidate article with the keyword classifier (`KeywordClassifier.classify_dataframe`) and country detector (`detect_countries`, `detect_oda_recipient_countries`) before sampling, using the same criteria `run_classify.py` uses to route articles to the BERT stage:

- **candidate** — `policy_actor == 1` OR (`kw_sdg_hits >= 2` AND mentions an ODA recipient country). Mirrors the actual BERT-eligible population.
- **borderline** — some SDG keyword hits or a country mention, but not both. The genuinely ambiguous cases that build coder calibration.
- **negative** — no keyword hits, no country mention. Kept as a smaller control group to confirm the true-negative rate.

Each year's sampling quota is drawn proportionally across strata (default 50% candidate / 30% borderline / 20% negative) instead of uniformly at random. This oversamples the population the classifier is actually scored against and gives coders enough boundary cases to build a consistent shared standard.

**Reporting caveat:** Because this is an enriched, not simple-random, sample, the prevalence of positive labels within it does **not** estimate corpus-wide prevalence and must not be reported as such. Precision/recall/F1 computed within the `candidate` stratum are valid estimates for that stratum specifically — which is also the exact population the trained classifiers are applied to, making it the right stratum for primary validation metrics.

**Current sample** (2026-07-09): n=595, seed=2025, overlap=150 (both coders label these first), stratum counts candidate=306/borderline=170/negative=119, evenly distributed across 2007–2023 (35/year).

---

## 5. Anticipated Peer Review Challenges and Responses

### Challenge 1: "Why Korean news? Korean public doesn't determine ODA allocations."

**Response:** Korea's ODA governance is domestically contested. KOICA and EDCF allocations are subject to parliamentary appropriations (ODA Act 2010; amended 2020) and the Prime Minister's ODA Policy Committee includes representatives from civil society and academia, both of whom are influenced by media discourse. Moreover, Korea's ODA has been shown to be responsive to domestic political economy factors (Park 2017; Kim & Kim 2013). The domestic audience model of foreign aid (Milner & Tingley 2013) does not require public opinion to directly determine allocations — media salience shapes bureaucratic and parliamentary attention, which affects allocation through multiple indirect channels.

### Challenge 2: "Your media variable is contaminated by domestic Korean news."

**Response:** This is precisely the concern that motivated the three-stage filtering design. The ODA-recipient-country filter (Stage 2) restricts articles to those mentioning countries in Korea's actual ODA recipient set, removing articles that discuss domestic Korean SDG issues without international development relevance. The translation + development-anchored embedding (Stage 3) further discriminates by requiring that the semantic content of the translated article align with ODA/development-specific anchor texts, not just the SDG topic in general. Residual contamination is acknowledged as a limitation but substantially reduced (candidate pool dropped 46% from the ODA country filter alone), and the robustness check using `policy_actor = 1` articles provides a high-precision upper bound.

### Challenge 3: "Machine translation introduces noise. Why not use a Korean-language classifier trained on Korean development text?"

**Response:** No Korean-language training dataset for ODA/development SDG classification exists to our knowledge. The OSDG and Aurora SDG classifiers, the two established SDG text classification systems, were trained on English-language documents (scientific publications and policy documents). Using Helsinki-NLP translation as a preprocessing step allows us to leverage these English-calibrated semantic spaces without requiring Korean-labeled training data. The keyword boost (+0.03 per hit, capped at 0.15) applied to the original Korean text provides a correction signal when key Korean development terms (KOICA, 공적개발원조, 수원국) appear, reducing dependence on translation quality for the most important articles. As a robustness check, the keyword-only specification does not require translation and serves as an alternative identification strategy.

### Challenge 4: "How do you handle articles relevant to multiple SDGs?"

**Response:** The classifier assigns a primary SDG label (highest cosine similarity + keyword boost) and a secondary SDG label (second highest, if cosine similarity ≥ 0.20). The main analysis uses primary SDG only; robustness checks with secondary-SDG-inclusive counting will be reported. This follows the approach of Sacchi et al. (2019) and the OSDG documentation, which recommend single-label classification for index construction to avoid double-counting. Articles with very close primary/secondary scores (within 0.05) could be flagged for multi-SDG treatment in sensitivity analysis.

### Challenge 5: "Your ODA SDG assignment via CRS crosswalk is imprecise."

**Response:** The Pincet et al. (2019) CRS–SDG crosswalk is the OECD's own recommended instrument for retrospective SDG tagging of ODA flows and has been used in peer-reviewed studies (Bhattacharya et al. 2018; Ohno et al. 2020; OECD DAC 2020 SDG Finance report). The alternative — project-level text classification of Korean ODA project descriptions — would require a Korean-language ODA text classifier that does not currently exist and introduces its own measurement uncertainty. The crosswalk approach is conservative (it maps sector-of-activity to SDG, which is well-defined) and consistent with how comparable studies have operationalized ODA-SDG alignment. Direct SDG tags (27.1% of projects) are used when available and take priority.

---

## 6. Data Disclosure and Replication Notes

| Item | Status |
|------|--------|
| BigKinds news data | Licensed dataset; raw files not public but available to researchers from BigKinds upon institutional request |
| ODA data source | `korea_oda data.xlsx` — Korea KOICA/EDCF official data; available from OECD CRS database and KOICA Open Data Portal |
| CRS–SDG crosswalk | Pincet et al. (2019), publicly available from OECD iLibrary |
| Translation model | `Helsinki-NLP/opus-mt-ko-en`, publicly available on HuggingFace (MIT License) |
| Embedding model | `intfloat/multilingual-e5-base`, publicly available on HuggingFace (MIT License) |
| Pipeline code | All preprocessing, classification, and panel construction code available in this repository |
| Keyword lists | In `pipeline/classify/keywords_ko.py`; reviewable and extensible |
| Country–ISO3 mapping | In `pipeline/reference/countries_ko.py`; based on UN Term Portal and MOFA Korea standards |
| Human annotation sample | Enriched stratified design (candidate/borderline/negative), see Section 4.4; generated by `pipeline/sample_for_labeling.py` |

---

*Last updated: 2026-07-09. Update this document when any pipeline parameter changes.*
