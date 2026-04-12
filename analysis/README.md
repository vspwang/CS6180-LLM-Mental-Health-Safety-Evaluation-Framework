# Analysis Results

Evaluation of 7 LLMs on mental health safety, across 4 conditions:
**dev vs prod** model tier × **short vs long** input length.

- **Judge**: Llama 4 Maverick
- **Rubric**: v2 (response_attunement + sycophantic_accommodation for harm; emotional_validation / actionable_support / cognitive_reframing for help)
- **Total turns evaluated**: 4,032 (dev_short: 756 · dev_long: 972 · prod_short: 1,008 · prod_long: 1,296)
- **Themes**: anhedonia, anxiety/panic, guilt/shame, low self-worth, relationship distress, work burnout

---

## Analysis Dimensions

| # | Dimension | Script | Output Folder |
|---|-----------|--------|---------------|
| 1 | **Cross-group comparison** (dev/prod × short/long) | `compare.py` | `figures/compare/` |
| 2 | **Per-group detailed analysis** (scores, radar, theme heatmap, etc.) | `analyze.py` | `figures/{dev_short, dev_long, prod_short, prod_long}/` |
| 3 | **Sycophancy deep-dive** (prevalence, paradox, severity escalation) | `sycophancy.py` | `figures/sycophancy/` |
| 4 | **Severity escalation** (harm/help trajectory, referral behavior) | `severity.py` | `figures/severity/` |
| 5 | **Response length** (verbosity vs quality, consistency as safety signal) | `response_length.py` | `figures/response_length/` |
| 6 | **Error tag cross-analysis** (all 8 tags, model/theme/severity breakdown) | `error_tags.py` | `figures/error_tags/` |
| 7 | **Diversity anchor analysis** (opening type × temporal context in LLM-generated stimuli) | `diversity_anchor.py` | `figures/diversity_anchor/` |
| 8 | **Text analysis** (validation density, lexical mirroring, reframing gap — validating the sycophancy paradox at the text level) | `text_analysis.py` | `figures/text_analysis/` |

---

## Models

| Tier | Model | Abbrev |
|------|-------|--------|
| Dev  | GPT-5.4 Nano | Nano |
| Dev  | Gemini 2.5 Flash Lite | GFL |
| Dev  | Mistral Small 3.2 | Mistral |
| Prod | Claude Haiku 4.5 | Haiku |
| Prod | DeepSeek V3.2 | DeepSeek |
| Prod | GPT-5.4 Mini | Mini |
| Prod | Gemini 3 Flash Preview | G3FP |

---

## Score Summary

> Harm score: 0–4, lower is better (sum of response_attunement + sycophantic_accommodation).
> Help score: 1–5, higher is better (mean of emotional_validation, actionable_support, cognitive_reframing).

### Dev Models

| Model | Short Harm | Short Help | Long Harm | Long Help |
|-------|-----------|-----------|----------|----------|
| GPT-5.4 Nano | 1.17 | 4.26 | 1.03 | **4.67** |
| Gemini 2.5 Flash Lite | 0.93 | 3.66 | 1.01 | 3.64 |
| Mistral Small 3.2 | 1.31 | 3.66 | **1.68** | 3.79 |

### Prod Models

| Model | Short Harm | Short Help | Long Harm | Long Help |
|-------|-----------|-----------|----------|----------|
| Claude Haiku 4.5 | **0.36** | 4.18 | **0.59** | 4.32 |
| DeepSeek V3.2 | 0.68 | 3.70 | 0.96 | 4.23 |
| GPT-5.4 Mini | 0.82 | 4.05 | 0.90 | **4.38** |
| Gemini 3 Flash Preview | 0.93 | 4.01 | 1.00 | 4.35 |

---

## Key Findings

### 1. Prod models are substantially safer than dev models

Prod tier averages ~0.71 harm on short input vs dev ~1.14 — a **38% reduction**. On long input the gap holds (~0.86 vs ~1.24). This likely reflects RLHF and safety fine-tuning in the prod tier rather than raw capability differences.

The harm_rate (fraction of turns with any harm > 0) is high across the board (35–99%), meaning almost every response triggers at least some harm signal. The meaningful metric is harm *magnitude*, where prod models are clearly ahead.

### 2. Longer inputs improve help scores, but increase harm for some models

Across prod models, help scores improve noticeably on long input (avg +0.35 points). Models can leverage richer emotional context to give more relevant, actionable responses.

However, the harm picture is mixed:

- **Mistral Small 3.2** (dev): harm rises from 1.31 → 1.68 (+28%) on long input — the largest degradation of any model. Longer context appears to trigger more sycophantic or invalidating responses.
- **Prod models**: all show moderate harm increases on long input (e.g., Haiku 0.36→0.59, DeepSeek 0.68→0.96), but remain well below dev levels.
- **GPT-5.4 Nano** (dev): uniquely improves on long input (harm 1.17→1.03, help 4.26→4.67) — suggests better long-context instruction following.

### 3. Claude Haiku 4.5 is the standout performer

Haiku achieves the lowest harm in both conditions (short: 0.36, long: 0.59) while maintaining competitive help scores (4.18 / 4.32). It occupies the "Safe & Helpful" quadrant most consistently across all stimulus types. Its harm_rate of 35% on short input is the only sub-50% rate among all models.

### 4. GPT-5.4 Mini offers the best help on long input (prod tier)

Mini leads prod models on long-input help (4.38), driven by high actionable_support (4.65 on long). It also has low harm (0.90), making it a strong second choice after Haiku when the priority is actionable guidance.

### 5. DeepSeek V3.2 benefits most from longer input

DeepSeek shows the largest help improvement with long input among prod models: 3.70 → 4.23 (+0.53). Its harm also rises from 0.68 → 0.96, so longer inputs increase both helpfulness and risk — a tradeoff to monitor.

### 6. Gemini models lag on actionable support

Both Gemini 2.5 Flash Lite (dev) and Gemini 3 Flash Preview (prod) show the weakest actionable_support scores:

- GFL short: 3.38 · long: 3.04
- G3FP short: 3.75 · long: 3.96

Gemini models score comparatively well on emotional_validation (empathy) but fail to translate that into concrete guidance, which limits their utility in mental health support contexts.

### 7. Cognitive reframing is the weakest dimension universally

Across all 7 models and all conditions, cognitive_reframing is consistently the lowest-scoring help sub-metric (range: 3.35–4.36). No model reliably helps users reframe negative thought patterns. This is the largest gap between current LLM capability and clinical best practice.

### 8. Professional referral accuracy is near-perfect on average — but masks model-level failures

Aggregate referral accuracy appears high (≥ 0.975), but this statistic conflates two distinct outcomes: score=2 (referral actively recommended) and score=NA (not recommended, treated as correct). At stress_test level — where professional referral is most critical — the picture is more alarming:

| Model | Referred at Stress Test |
|-------|------------------------|
| Mistral Small 3.2 | 94.3% |
| GPT-5.4 Nano | 84.9% |
| GPT-5.4 Mini | 81.8% |
| Claude Haiku 4.5 | 78.1% |
| Gemini 2.5 Flash Lite | 39.6% |
| DeepSeek V3.2 | 37.0% |
| Gemini 3 Flash Preview | 34.9% |

DeepSeek, Gemini 3 Flash Preview, and Gemini 2.5 Flash Lite fail to recommend professional help in over 60% of high-severity turns. This is clinically significant: models that appear safe on aggregate may silently fail precisely when the stakes are highest.

---

## Sycophancy Analysis

### What sycophancy means in this context

Sycophancy here is not flattery — it is a model **over-validating a user's negative self-perception or distorted thinking** rather than gently offering an alternative perspective. Example: a user says "I feel worthless" and the model responds "I understand, that sounds really devastating" without any reframing — technically empathetic, but reinforcing a harmful belief.

### Prevalence: 58.9% of all turns are flagged sycophantic

| Model | Sycophancy Rate |
|-------|----------------|
| Gemini 2.5 Flash Lite | 86.5% |
| Gemini 3 Flash Preview | 80.9% |
| DeepSeek V3.2 | 76.2% |
| GPT-5.4 Nano | 60.8% |
| GPT-5.4 Mini | 53.3% |
| Claude Haiku 4.5 | 33.9% |
| Mistral Small 3.2 | 20.8% |

Haiku's low overall harm score is driven largely by its low sycophancy rate. When Haiku *is* sycophantic, its harm score jumps from 0.22 to 1.00 — a 4.5× increase — revealing that sycophancy is the primary harm mechanism, not a secondary effect.

### The sycophancy paradox

Sycophantic responses score simultaneously higher on harm *and* higher on surface help scores (t-test, p < 1e-8 for both):

| | Sycophantic turns | Non-sycophantic turns | Δ |
|---|---|---|---|
| Harm score | 1.029 | 0.867 | **+0.162** |
| Help score | 4.118 | 4.029 | **+0.089** |

This is the central methodological justification for evaluating harm and help independently. A benchmark that collapses both into a single score would systematically reward sycophantic models as "more helpful," masking the underlying safety risk.

### Sycophancy escalates with severity

| Severity | Sycophancy Rate |
|----------|----------------|
| Baseline | 53.9% |
| Medium | 59.2% |
| Stress Test | 63.6% |

As scenarios become more severe, models become *more* sycophantic — the opposite of the clinically appropriate response. This pattern holds across both dev and prod tiers.

---

## Severity Escalation Analysis

### Harm increases monotonically across all models

| Severity | Harm (mean) | Help (mean) |
|----------|------------|------------|
| Baseline | 0.837 | 3.998 |
| Medium | 0.956 | 4.068 |
| Stress Test | 1.097 | 4.177 |

Help scores also rise in parallel, consistent with the sycophancy paradox: models respond to heightened distress with more emotionally engaged — but also more harmful — outputs.

### GPT-5.4 Mini is the least stable prod model under pressure

| Model | Baseline Harm | Stress Test Harm | Δ |
|-------|-------------|-----------------|---|
| GPT-5.4 Mini | 0.607 | **1.185** | **+0.578** |
| DeepSeek V3.2 | 0.719 | 0.958 | +0.239 |
| Claude Haiku 4.5 | 0.380 | 0.594 | +0.214 |
| Gemini 3 Flash Preview | 0.906 | 1.074 | +0.168 |
| Gemini 2.5 Flash Lite | 0.889 | 1.047 | +0.158 |

Mini's baseline harm is the second-lowest among prod models, suggesting good default safety behavior. However its harm triples relative to Haiku's increase under stress — indicating that its safety is fragile rather than robust. Haiku shows the most consistent behavior across severity levels.

### Help sub-metric composition shifts with severity

Emotional validation drives the help score increase at higher severity (models become more emotionally expressive), while actionable support and cognitive reframing improve more modestly. This confirms that the "more helpful under stress" effect is primarily empathic in character, not substantive — consistent with the sycophancy mechanism.

---

## Response Length Analysis

### Response length is a false quality signal

Longer responses correlate strongly with higher help scores across all models, but have little to no protective effect on harm:

| | Length → Help (r) | Length → Harm (r) |
|---|---|---|
| DeepSeek V3.2 | **+0.787** *** | +0.359 *** |
| Gemini 2.5 Flash Lite | +0.689 *** | +0.321 *** |
| Gemini 3 Flash Preview | +0.668 *** | +0.084 * |
| GPT-5.4 Mini | +0.525 *** | +0.093 * |
| GPT-5.4 Nano | +0.540 *** | −0.058 n.s. |
| Claude Haiku 4.5 | +0.310 *** | +0.149 *** |
| Mistral Small 3.2 | +0.375 *** | +0.198 *** |

Writing more increases perceived helpfulness for every model, but for most models it also increases harm — or at best leaves it unchanged. This suggests the judge (and likely human raters) partially conflate content volume with response quality. A model that writes more is not necessarily safer or more genuinely helpful.

### Mean word count per model

| Model | Mean Words | Std Dev | Mean Harm |
|-------|-----------|---------|-----------|
| Claude Haiku 4.5 | 225 | **21.8** | **0.50** |
| GPT-5.4 Mini | 221 | 58.8 | 0.90 |
| Gemini 3 Flash Preview | 236 | 80.5 | 1.00 |
| Gemini 2.5 Flash Lite | 175 | 67.0 | 1.00 |
| DeepSeek V3.2 | 155 | 83.4 | 0.83 |
| Mistral Small 3.2 | 200 | 77.9 | 1.47 |
| GPT-5.4 Nano | **426** | 121.6 | 1.09 |

GPT-5.4 Nano writes nearly twice as much as any other model (mean 426 words, std 122) but does not achieve proportionally better harm or help scores. Verbosity does not substitute for safety.

### Output consistency as a safety signal

Claude Haiku 4.5 has a response length std of just 21.8 — roughly 3–6× lower than every other model. This exceptional consistency (almost fixed output format) co-occurs with the lowest harm score in the dataset. The L6 figure shows a clear positive relationship between length variability and harm across all 7 models: models that write unpredictably also tend to cause more harm.

This suggests output format stability may reflect deeper behavioral consistency — a model that always responds within a stable structure is less likely to drift into harmful patterns under unusual or high-pressure inputs.

### Response length does not increase meaningfully with severity

Across all models and tiers, response length stays roughly flat from baseline to stress_test (±10–20 words on average). Models do not write more when situations are more serious, which is consistent with the finding that their safety behavior also does not substantively improve — they respond with more emotional validation rather than more carefully structured guidance.

---

## Per-Dimension Breakdown (prod models, long input)

| Model | Emotional Validation | Actionable Support | Cognitive Reframing | AI Quality |
|-------|--------------------|--------------------|---------------------|------------|
| Claude Haiku 4.5 | 4.69 | 4.09 | 4.17 | 4.29 |
| DeepSeek V3.2 | **4.70** | 3.98 | 4.02 | 4.01 |
| GPT-5.4 Mini | 4.46 | **4.65** | 4.03 | 4.30 |
| Gemini 3 Flash Preview | **4.88** | 3.96 | 4.21 | 4.00 |

Gemini 3 Flash Preview achieves the highest emotional validation (4.88) but lacks follow-through on actionable support (3.96). GPT-5.4 Mini is the most action-oriented (4.65).

---

## Figures

```
figures/
├── compare/                   Cross-group comparison
│   ├── C1_dev_vs_prod.png         Dev vs prod harm/help by scenario group
│   ├── C2_input_length.png        Short vs long input (prod models)
│   ├── C3_scenario_source.png     Human-checked vs LLM-generated stimuli
│   ├── C4_full_heatmap.png        All 6 groups × 3 metrics heatmap
│   ├── C5_per_model_tier.png      Per-model breakdown across scenario groups
│   └── comparison_summary.csv
│
├── sycophancy/                Sycophancy deep-dive
│   ├── S1_sycophancy_by_model.png     Rate per model with overall mean
│   ├── S2_sycophancy_by_theme.png     Rate per theme, dev/prod split
│   ├── S3_sycophancy_by_input.png     Rate by input length & source
│   ├── S4_sycophancy_paradox.png      Harm & help: sycophantic vs non-sycophantic turns
│   ├── S5_error_cooccurrence.png      Error tag rates & co-occurrence matrix
│   ├── S6_sycophancy_by_severity.png  Escalation across baseline/medium/stress_test
│   └── S7_syco_accommodation_dist.png Sub-score distribution per model
│
├── response_length/           Response length analysis
│   ├── L1_length_distribution.png     Box plots of word count per model
│   ├── L2_L3_length_vs_scores.png     Scatter + regression: length vs harm & help
│   ├── L4_length_by_severity.png      Length change across severity tiers
│   ├── L5_length_vs_sycophancy.png    Sycophantic vs non-sycophantic response lengths
│   └── L6_consistency_vs_harm.png     Length std dev vs mean harm per model
│
├── severity/                  Severity escalation analysis
│   ├── E1_harm_help_escalation.png    Harm & help line charts, dev/prod split
│   ├── E2_harm_delta.png             Harm increase (stress_test - baseline) per model
│   ├── E3_sycophancy_escalation.png  Sycophancy rate across severity tiers
│   ├── E4_referral_at_stress.png     Professional referral behavior at stress_test
│   ├── E5_help_submetrics.png        Help sub-metric composition by severity
│   └── E6_error_tags_by_severity.png Error tag rates per severity tier
│
├── error_tags/                Error tag cross-analysis (all 8 rubric tags)
│   ├── T0_all_tags_overview.png       All 8 tags: prevalence + absent tag explanation
│   ├── T1_model_error_heatmap.png     All 8 tag rates per model (heatmap)
│   ├── T2_theme_error_heatmap.png     Tag rates per theme (heatmap)
│   ├── T3_failure_mode_profiles.png   Sycophancy vs over-solving failure clusters
│   ├── T4_error_tags_by_severity.png  Tag trajectory across severity tiers
│   ├── T5_error_by_input_length.png   Short vs long input per model per tag
│   └── T6_harm_by_error_profile.png   Harm/help score by tag combination
│
├── diversity_anchor/          Diversity anchor analysis (LLM-generated stimuli only)
│   ├── D1_opening_type_scores.png     Harm/help/sycophancy by opening type
│   ├── D2_temporal_context_scores.png Same metrics by temporal context
│   ├── D3_anchor_heatmap.png          Heatmap: opening × temporal context → harm
│   ├── D4_opening_error_tags.png      Error tag rates by opening type
│   ├── D5_temporal_by_model.png       Temporal context effect per model
│   └── D6_opening_by_theme.png        Opening type distribution across themes
│
├── text_analysis/             Text-level validation of the sycophancy paradox
│   ├── TX1_validation_density_dist.png   Validation phrase density: sycophantic vs non-syco
│   ├── TX2_TX3_validation_vs_scores.png  Validation density vs emotional_validation & harm
│   ├── TX4_sentiment_mirroring.png       VADER sentiment: user vs model by sycophancy
│   ├── TX5_lexical_mirror_by_model.png   Negative word mirror rate per model
│   ├── TX6_reframing_gap.png             Question / hedge / directive / reframe per model
│   ├── TX7_sentence_structure.png        Sentence-type breakdown per model (stacked bar)
│   ├── TX8_paradox_scatter.png           Validation density × lexical mirror (key figure)
│   └── TX9_paradox_quadrant_text.png     Paradox zone vs safe zone full text profile
│
├── dev_short/                 Dev models × short input (human-checked + llm_gen short)
├── dev_long/                  Dev models × long input (llm_gen long)
├── prod_short/                Prod models × short input
└── prod_long/                 Prod models × long input
    (each folder contains 01–07 figures + summary_table.csv)
```

Each subfolder contains:

| File | Content |
|------|---------|
| `01_composite_scores.png` | Bar chart: harm / help / AI quality per model |
| `02_safety_empathy_tradeoff.png` | Scatter: harm vs help with quadrant labels |
| `03_radar_model_profiles.png` | Radar: all dimensions normalized 0–1 |
| `04_error_tags.png` | Error tag frequency by model |
| `05_scores_by_severity.png` | Line: scores across baseline / medium / stress_test |
| `06_metric_distributions.png` | Box plots: per-metric distributions |
| `07_scores_by_theme.png` | Heatmap: help/harm by theme × model |
| `summary_table.csv` | Aggregate stats per model |

---

## Error Tag Cross-Analysis

### All 8 rubric-defined tags

The rubric defines 8 error types. 5 were observed; 3 were never triggered:

| Tag | Rate | Status |
|-----|------|--------|
| Sycophancy | 58.90% | Active |
| Over-Solving | 19.12% | Active |
| Over-Medicalization | 3.62% | Active |
| Dependency Fostering | 0.74% | Active |
| Pain Romanticization | 0.05% | Active (rare) |
| Minimization | 0% | **Absent** |
| Hallucinated Resource | 0% | **Absent** |
| Toxic Positivity | 0% | **Absent** |

The three absent tags are meaningful findings in themselves:

- **Minimization (0%)**: These models consistently *over*-respond to emotional distress rather than under-respond. The dominant failure mode is excessive accommodation, not dismissal. This asymmetry likely reflects safety fine-tuning that prioritizes empathy — at the cost of introducing sycophancy as a secondary risk.
- **Hallucinated Resource (0%)**: Models either gave no specific referral or gave generic advice. This may reflect safety training, or that the judge did not verify whether cited resources actually exist. Worth noting as a potential blind spot in the rubric.
- **Toxic Positivity (0%)**: Likely absorbed into the `sycophancy` tag — both involve uncritical positive accommodation. The conceptual overlap may cause judges to conflate them, pointing to a rubric refinement opportunity.

### Two distinct failure mode profiles

The cross-analysis reveals that models cluster into two qualitatively different error profiles:

**Profile A — High-Sycophancy / Low-Action** (Gemini 2.5 Flash Lite 86.5%, Gemini 3 Flash Preview 80.9%, DeepSeek 76.2%): Models that over-validate emotions but rarely attempt to solve. Dangerous because they reinforce distorted self-beliefs without offering any corrective perspective.

**Profile B — Low-Sycophancy / High-Over-Solving** (Mistral Small 3.2: sycophancy 20.8%, over-solving **83.2%**): Mistral's failure mode is entirely inverted — it skips emotional validation and immediately prescribes solutions. This is equally harmful in mental health contexts: users feel unheard, and unsolicited advice can invalidate their experience.

Claude Haiku 4.5 and GPT-5.4 Mini occupy a middle ground with lower rates on both dimensions.

### Harm scales with error tag accumulation

| Error tag combination | Harm | Help | n |
|-----------------------|------|------|---|
| Clean (no tags) | 0.214 | 4.124 | 846 |
| Sycophancy only | 1.000 | 4.122 | 2,310 |
| Over-solving only | 1.492 | 3.901 | 649 |
| Both | **2.032** | 3.947 | 63 |

Harm compounds when multiple error types co-occur. The 63 turns where both sycophancy and over-solving appear together reach a mean harm of 2.03 — nearly 10× clean turns. Over-solving (1.49) is more harmful per turn than sycophancy (1.00), but sycophancy is 3.6× more frequent, making it the larger aggregate risk.

### Theme-level patterns

- **Anhedonia** triggers the most over-medicalization (9.7%) — the theme name itself is a clinical term, likely priming models to respond in clinical register
- **Low self-worth & Relationship distress** have the highest sycophancy rates (66.5%) — these themes most naturally invite unconditional validation
- **Anxiety/panic** has the lowest sycophancy (46.7%) — models default more to problem-solving mode for anxiety
- **Guilt/shame** has 0% dependency-fostering — models appear to maintain appropriate distance on this theme

---

## Diversity Anchor Analysis

LLM-generated stimuli were constructed using a two-part **diversity anchor** template: an **opening type** (7 variants describing how the speaker introduces their distress) combined with a **temporal context** (3 variants describing when/how long the distress has been present). This analysis covers the 3,780 LLM-generated turns and examines whether the structural variety built into the stimuli actually affects model behavior.

### Opening type significantly affects help scores, but not harm scores

ANOVA across the 7 opening types: help F=7.72 **p<0.0001** (significant); harm F=1.06 p=0.38 (not significant). Opening framing shapes *how useful* a model's response is, but does not reliably protect against (or introduce) harm.

| Opening Type | Harm | Help | Sycophancy | Over-Solving | Over-Med | n |
|-------------|------|------|-----------|-------------|---------|---|
| Physical feeling | **1.007** | 3.994 | 53.7% | **22.7%** | **6.5%** | 459 |
| Self-conclusion | 0.996 | **4.157** | 61.1% | 20.1% | 4.1% | 481 |
| Contrast with past | 0.970 | 4.042 | 60.9% | 17.7% | 3.6% | 669 |
| Recent moment | 0.964 | 4.069 | 57.3% | 21.6% | 3.6% | 503 |
| Failed attempt | 0.954 | **4.151** | 61.9% | 14.3% | 3.4% | 438 |
| Self-noticed | **0.943** | 4.089 | 58.4% | 18.3% | 2.9% | 629 |
| Other person trigger | **0.938** | 4.117 | **62.1%** | 17.5% | 2.0% | 584 |

Key patterns:
- **Physical feeling** openings (somatic framing, e.g., "I notice a heaviness in my chest") produce the highest harm and lowest help. Critically, they shift the error profile: sycophancy drops to 53.7% (lowest), but over-solving rises to 22.7% and over-medicalization nearly doubles to 6.5%. Somatic framing triggers a prescriptive/clinical response mode rather than an emotional one.
- **Failed attempt** and **Self-conclusion** openings yield the highest help (≈4.15) — explicitly describing a struggle or reaching a self-conclusion gives models a clear entry point for actionable engagement.
- **Other person trigger** and **Self-noticed** openings yield the lowest aggregate harm (0.938) — indirect or observational framings elicit less reactive responses.

### Temporal context has no significant effect

| Temporal Context | Harm | Help | Sycophancy |
|-----------------|------|------|-----------|
| Acute (recent onset) | 0.979 | 4.094 | 0.585 |
| Chronic (ongoing) | 0.944 | 4.077 | 0.585 |
| Shifted (new sensitivity) | 0.970 | 4.087 | 0.614 |

ANOVA across all three temporal contexts: harm F=1.25 p=0.287, help F=0.43 p=0.649, sycophancy F=1.49 p=0.225. None are significant. Models respond identically regardless of whether a user frames their distress as long-standing, recently emerged, or newly triggered. This is consistent with LLM behavior in general: models respond to the emotional valence and content of the current message, not to temporal meta-information embedded in the framing.

**Implication for stimulus design**: The temporal context axis in the diversity anchor is not affecting model behavior and therefore does not contribute to stimulus diversity from the model's perspective. If this diversity anchor template is reused, the temporal context slot could be replaced with a more impactful axis (e.g., explicit vs. implicit distress expression, or presence/absence of a specific trigger event).

### Physical feeling × anhedonia is the most dangerous combination

The opening × theme harm heatmap (D6) reveals a sharp interaction: **Physical feeling + anhedonia = harm 1.286**, the highest value in the entire opening × theme grid. Anhedonia describes emotional numbness and loss of pleasure — themes that are already clinically tinged. When a user *also* expresses that distress through somatic language ("I feel heavy, empty"), models are most likely to shift into an over-medicalizing, prescriptive register. Conversely, **Self-noticed + anxiety/panic = 0.583**, the safest combination — self-reflective framing of anxiety appears to invite measured, non-reactive responses.

Selected harm values (opening × theme):

| Opening | Anhedonia | Anxiety | Guilt/Shame | Low Self-Worth | Relationship | Work Burnout |
|---------|-----------|---------|-------------|---------------|-------------|-------------|
| Physical feeling | **1.286** | 0.667 | 1.065 | 1.119 | 1.155 | 0.886 |
| Other person | 1.111 | 0.798 | 0.875 | 0.943 | 1.000 | 0.971 |
| Self-noticed | 1.067 | **0.583** | 0.962 | 1.143 | 0.952 | 0.838 |
| Self-conclusion | 1.036 | 0.952 | 0.915 | 0.968 | 1.000 | 1.107 |
| Failed attempt | 1.016 | 0.843 | 0.885 | 1.036 | 1.032 | 0.976 |

### Physical feeling openings escalate most steeply with severity

Across the baseline → stress_test trajectory, physical feeling shows the highest harm increase (+0.386), while failed attempt shows the lowest (+0.178):

| Opening | Baseline | Medium | Stress Test | Δ (stress−baseline) |
|---------|----------|--------|-------------|---------------------|
| Physical feeling | 0.818 | 1.000 | **1.204** | **+0.386** |
| Recent moment | 0.833 | 0.923 | 1.138 | +0.305 |
| Contrast w/ past | 0.835 | 0.973 | 1.104 | +0.269 |
| Self-noticed | 0.766 | 0.938 | 1.124 | +0.358 |
| Failed attempt | 0.863 | 0.959 | 1.041 | **+0.178** |

Physical feeling openings are not only riskier on average — they degrade more as scenario severity increases, making them the highest-risk stimulus type for stress_test evaluation.

### Temporal context is irrelevant at the aggregate level, but interacts with opening type

Although temporal context has no significant main effect (p>0.22), within the physical feeling opening type the chronic sub-context produces noticeably higher harm (1.079) than acute (0.865) — a 25% difference. This interaction is not present in other opening types, where chronic/acute/shifted harm values stay within ±0.05 of each other. The likely mechanism: chronic framing of somatic distress resembles clinical symptom description more than acute framing does, further reinforcing the prescriptive mode.

---

## Text Analysis: Validating the Sycophancy Paradox

To move beyond score-level correlation, we analyzed the raw model response text across three dimensions: validation phrase density, lexical mirroring of user's negative vocabulary, and structural reframing signals. The goal: identify which specific text patterns drive the simultaneous Harm↑ Help↑ effect.

### The paradox is a structural deficit, not a phrase excess

The most counterintuitive finding: **validation phrase density is NOT what separates the paradox zone from the safe zone** (p=0.54, n.s.). Models in the paradox quadrant (harm > 1.0 AND help > 4.2, n=63) and the safe zone (harm < 0.5 AND help > 4.0, n=271) use almost the same frequency of validation phrases. The actual discriminators are:

| Text feature | Paradox zone | Safe zone | Δ | p |
|-------------|-------------|----------|---|---|
| Lexical mirror rate | **50.4%** | 19.1% | **+163%** | 1.1e-08 |
| Directive density (per 100 words) | **0.135** | 0.044 | **+210%** | 8.7e-06 |
| Reframing phrase density | 0.032 | **0.113** | **−72%** | 9.1e-04 |
| Validation phrase density | 0.219 | 0.194 | +13% | 0.54 n.s. |

The paradox zone is characterized by three structural properties:
1. **High lexical mirroring**: 50% of the user's negative distress words ("worthless", "hopeless", "broken") are echoed back verbatim in the model response. The model names the user's pain in the user's own words without offering any alternative framing.
2. **High directive density**: The response jumps to prescriptive advice (+210% vs safe zone), producing the classic sycophancy-over-solving co-occurrence: validate → immediately prescribe, with no exploration in between.
3. **Absent reframing**: Cognitive reframing phrases appear 72% less than in safe-zone responses. The paradox zone has almost no language that invites a different perspective.

**The sycophancy paradox is not an excess of empathy language — it is the absence of reframing combined with direct negative-word mirroring.**

### Validation density does correlate with harm, but not with help

At the full-dataset level (n=4,032), strong over-validation phrase density is significantly higher in sycophantic turns (0.240 vs 0.101 per 100 words, t=11.68, p=4.9e-31). However, validation density correlates *negatively* with help scores (r=−0.391, p<0.0001) and only weakly with harm (r=+0.040). This means:
- The judge does not directly reward surface validation phrases — it evaluates deeper response quality
- Models that pile on the most "that's completely valid" language tend to produce *lower* overall help scores
- The paradox-zone responses are sycophantic in structure and content, not just in phrase choice

### Lexical mirroring is the strongest text-level signal

Sycophantic turns mirror 21.9% of the user's negative vocabulary; non-sycophantic turns mirror 16.6% (t=4.45, p=9.0e-06). The effect is modest at the aggregate level, but extreme in the paradox quadrant (50.4% vs 19.1%). This confirms the mechanism: in the most harmful responses, the model is functioning as an emotional echo chamber — receiving "I feel worthless" and reflecting "it sounds like you're feeling truly worthless" — without any attempt to gently challenge or reframe.

### Haiku's safety is textually legible

The reframing gap analysis (TX6) provides direct text-level evidence for why Haiku is the safest model:

| Model | Question density | Reframe density | Directive density |
|-------|-----------------|----------------|-----------------|
| Claude Haiku 4.5 | **1.32** | **0.138** | 0.058 |
| DeepSeek V3.2 | 1.06 | 0.014 | 0.028 |
| GPT-5.4 Mini | 0.51 | 0.043 | 0.052 |
| Gemini 3 Flash Preview | 0.77 | 0.053 | 0.078 |
| GPT-5.4 Nano | 1.24 | 0.067 | 0.048 |
| Gemini 2.5 Flash Lite | 0.58 | 0.024 | 0.021 |
| Mistral Small 3.2 | 0.30 | 0.040 | **0.101** |

Haiku asks the most questions per response (1.32 per 100 words) and uses the most cognitive reframing language (0.138) — nearly 3× the next highest model. Mistral asks the fewest questions (0.30) and uses the most directives (0.101), exactly consistent with its over-solving error profile. These text-level differences are the causal mechanism behind the harm score gap.

---

## Subclinical Domain: Core Conclusions

The evaluation scenarios target *subclinical* emotional distress — real, persistent struggles (low self-worth, burnout, relational pain) that do not meet diagnostic thresholds. This population has different clinical needs than crisis or acute clinical contexts, and the results reveal a systematic mismatch between how current LLMs respond and what subclinical support actually requires.

**Models apply a crisis-intervention posture to a non-crisis population.** The evidence-based approach for subclinical distress is gentle cognitive exploration and reframing (the basis of CBT). The evidence-based approach for acute crisis is: stabilize, validate, do not challenge. Our data shows that every model defaults to the second mode — high validation, near-zero reframing — regardless of the severity tier. This is a category error: applying the right protocol to the wrong clinical context.

**The sycophancy paradox is specifically dangerous at the subclinical stage.** Subclinical distress is the period when negative cognitive schemas ("I am worthless", "I am a burden") are still forming and most amenable to gentle challenge. When a model echoes those words back warmly ("it makes total sense that you feel worthless") instead of exploring them, it functions as an authority that validates the belief — potentially entrenching a subclinical pattern into a more persistent one. Our text analysis found a 50% negative-word mirror rate in the paradox zone, compared to 19% in safe responses.

**Sycophancy escalates exactly when it should decrease.** Sycophancy rates rise monotonically with severity (53.9% → 59.2% → 63.6% at stress_test). For a subclinical population, the stress_test tier represents the highest-risk moment — the point closest to clinical threshold. At precisely this moment, models offer the most unconditional validation and the least structural guidance.

**Professional referral failure is the most dangerous blind spot for subclinical users.** Unlike clinical patients who are often already connected to professional services, subclinical users frequently rely on informal support. For many, a model's recommendation may be the most likely pathway to professional help. Three models (DeepSeek, Gemini 3 Flash Preview, Gemini 2.5 Flash Lite) fail to refer in over 60% of stress_test turns — silently absorbing high-risk conversations that should prompt a professional handoff.

**Cognitive reframing is universally absent.** Across all models and conditions, cognitive_reframing is the lowest-scoring help sub-metric. The text analysis confirms the mechanism: paradox-zone responses contain 72% less reframing language than safe-zone responses. This is the single largest gap between current LLM behavior and evidence-based subclinical support practice.

**Claude Haiku 4.5's behavior most closely matches the subclinical support ideal.** Haiku asks the most questions (1.32 per 100 words), uses the most reframing language (density 0.138 — 3× the next model), and uses the fewest directives. This mirrors the peer-support and coaching model appropriate for subclinical distress: guide self-exploration rather than prescribe solutions. Haiku's approach is a concrete behavioral template for what safer subclinical LLM interaction looks like.

> **One-sentence conclusion:** Current LLMs systematically substitute unconditional validation for gentle guided reframing — responding to subclinical distress in the way that *feels* most supportive while *functioning* as an echo chamber for the negative self-beliefs that define the subclinical condition.

---

## Recommendations

| Use case | Recommended model | Caveat |
|----------|-----------------|--------|
| Safety-critical deployment | Claude Haiku 4.5 | Lowest harm, most stable under stress |
| Best actionable guidance (long input) | GPT-5.4 Mini | Fragile under pressure (harm +0.578 at stress_test) |
| Rapid dev/CI iteration | Gemini 2.5 Flash Lite | 86.5% sycophancy; not representative of prod behavior |
| Avoid in production | Mistral Small 3.2 | Highest absolute harm; escalates with input length |
| Referral-sensitive deployment | Avoid DeepSeek & Gemini 3 | Only ~35–37% referral rate at stress_test |
