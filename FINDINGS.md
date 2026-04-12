# LLM Mental Health Safety Evaluation — Findings

Evaluation of 7 LLMs on subclinical mental health support quality, using a dual-axis rubric that independently scores **harm** (response_attunement + sycophantic_accommodation, 0–4) and **help** (emotional_validation + actionable_support + cognitive_reframing, 1–5). Judge model: Llama 4 Maverick. Total turns evaluated: 4,032 across 6 mental health themes and 3 severity tiers.

---

## Models and Setup

Seven models were evaluated across two tiers. **Dev tier** (lower cost, faster iteration): GPT-5.4 Nano, Gemini 2.5 Flash Lite, Mistral Small 3.2. **Prod tier** (full-capability): Claude Haiku 4.5, DeepSeek V3.2, GPT-5.4 Mini, Gemini 3 Flash Preview.

Each model responded to stimuli drawn from two sources: 14 human-authored scenarios and 108 LLM-generated scenarios structured with a two-part diversity anchor (opening type × temporal context). Scenarios span three severity tiers — baseline, medium, and stress_test — all subclinical (no suicidality, no acute crisis).

---

## Score Overview

Prod models average ~0.71–0.86 harm vs dev models at ~1.14–1.24 — a roughly 38% reduction, likely reflecting safety fine-tuning rather than raw capability. Claude Haiku 4.5 achieves the lowest harm across both input lengths (0.36 short, 0.59 long) while maintaining competitive help scores. GPT-5.4 Nano writes nearly twice as many words as any other model (mean 426 words) but does not achieve proportionally better outcomes. Mistral Small 3.2 has the highest absolute harm and the only model whose harm *increases* with input length (+28%).

| Model | Harm (short) | Help (short) | Harm (long) | Help (long) |
|-------|-------------|-------------|------------|------------|
| Claude Haiku 4.5 | **0.36** | 4.18 | **0.59** | 4.32 |
| GPT-5.4 Mini | 0.82 | 4.05 | 0.90 | **4.38** |
| DeepSeek V3.2 | 0.68 | 3.70 | 0.96 | 4.23 |
| Gemini 3 Flash Preview | 0.93 | 4.01 | 1.00 | 4.35 |
| GPT-5.4 Nano | 1.17 | 4.26 | 1.03 | 4.67 |
| Gemini 2.5 Flash Lite | 0.93 | 3.66 | 1.01 | 3.64 |
| Mistral Small 3.2 | 1.31 | 3.66 | **1.68** | 3.79 |

---

## The Sycophancy Paradox

**What sycophancy means here.** In this rubric, sycophancy is not flattery — it is a model over-validating a user's negative self-perception without offering any reframe. When a user says "I feel worthless" and a model responds "that sounds truly devastating and your feelings are completely valid" without any gentle alternative perspective, it scores as sycophantic regardless of how warm it sounds.

**Prevalence is high across the board.** 58.9% of all turns are flagged sycophantic. Gemini 2.5 Flash Lite leads at 86.5%; Mistral is lowest at 20.8% (but has the highest over-solving rate instead). Claude Haiku 4.5 is the only model below 35%.

**The paradox.** Sycophantic turns score simultaneously higher on harm AND higher on help (t-test, p < 1e-8 for both). Harm: 1.029 vs 0.867 (+0.162). Help: 4.118 vs 4.029 (+0.089). A benchmark that collapses harm and help into a single score would systematically reward sycophantic models — precisely because sycophancy inflates surface helpfulness while introducing underlying harm. This is the core methodological justification for evaluating the two dimensions independently.

**Sycophancy escalates with severity.** Sycophancy rates rise from 53.9% (baseline) to 59.2% (medium) to 63.6% (stress_test). For a subclinical population, stress_test represents the highest-risk moment — and models respond with the most unconditional validation at exactly that point.

---

## Text-Level Mechanism: What Drives the Paradox

To move beyond score correlations, we extracted text features from all 4,032 responses: validation phrase density, negative-word lexical mirroring, question density, hedge ratio, directive density, and cognitive reframing markers.

**The paradox is a structural deficit, not a phrase excess.** The intuitive hypothesis — that paradox-zone responses pile on more validation phrases — is wrong. Validation phrase density is not significantly different between the paradox zone (harm > 1.0 AND help > 4.2) and the safe zone (harm < 0.5 AND help > 4.0, p = 0.54). The actual discriminators are:

The paradox zone has a **50.4% lexical mirror rate** vs 19.1% in the safe zone (+163%, p = 1.1e-08). This means half of the negative words a user uses — "worthless", "hopeless", "broken" — appear verbatim in the model response, functioning as an emotional echo chamber rather than a supportive reframe.

The paradox zone has **2.1× more directive language** than the safe zone (p = 8.7e-06). After mirroring the user's distress, the model skips any exploration and jumps straight to prescriptive advice — the classic sycophancy-over-solving co-occurrence.

The paradox zone has **72% less cognitive reframing language** than the safe zone (p = 9.1e-04). The language that would gently offer alternative perspectives ("have you considered", "another way to see this") is almost entirely absent.

**Why Haiku is different, in text.** Haiku asks the most questions per response (1.32 per 100 words, vs the next model at 1.24 and the lowest at 0.30 for Mistral). It uses the most cognitive reframing language (density 0.138 — nearly 3× the next model). It uses the fewest directives. This behavioral profile — explore before prescribe, reframe rather than mirror — is what makes its harm score the lowest in the dataset. The text makes the mechanism legible.

---

## Error Modes and Failure Profiles

The rubric defines 8 error tags. Five were observed; three (minimization, hallucinated resource, toxic positivity) were never triggered. The absence of minimization is meaningful: these models consistently *over-respond* to emotional distress rather than dismiss it. The dominant risk is excessive accommodation, not coldness.

**Two distinct failure profiles emerge.** Profile A — High Sycophancy / Low Action — characterizes Gemini 2.5 Flash Lite (86.5%), Gemini 3 Flash Preview (80.9%), and DeepSeek (76.2%). These models over-validate emotions without attempting to guide or reframe. Profile B — Low Sycophancy / High Over-Solving — is Mistral exclusively (sycophancy 20.8%, over-solving 83.2%). Mistral skips emotional validation entirely and immediately prescribes solutions — equally harmful in a support context, for the opposite reason.

**Harm compounds when profiles overlap.** Clean turns (no tags) average 0.21 harm. Sycophancy-only: 1.00. Over-solving-only: 1.49. Both simultaneously: 2.03 — nearly 10× clean turns. Over-solving causes more harm per turn; sycophancy is 3.6× more frequent, making it the larger aggregate risk.

---

## Severity and Referral Behavior

Harm rises monotonically across all models from baseline to stress_test. Help rises too — consistent with the paradox: models respond to heightened distress with more emotionally engaged responses that are simultaneously more harmful. GPT-5.4 Mini shows the largest harm increase (+0.578), making it the most fragile prod model under pressure despite its strong baseline.

**Professional referral at stress_test is the most clinically critical metric.** Three models — Gemini 3 Flash Preview (34.9%), DeepSeek (37.0%), Gemini 2.5 Flash Lite (39.6%) — fail to recommend professional help in over 60% of their highest-severity turns. For subclinical users who may not seek professional help independently, a model's referral suggestion may be the only prompt toward professional support they receive. Models that appear safe on aggregate scores may silently fail at the highest-stakes moment.

---

## Subclinical Domain: What This Means

Subclinical emotional distress is the stage when negative cognitive patterns are still forming — most amenable to gentle challenge and reframing. The evidence-based intervention for this population is exploratory, cognitively guided support (the basis of CBT and peer coaching), not the stabilize-and-validate protocol appropriate for acute crisis.

Every model in this evaluation defaults to the crisis protocol regardless of severity tier. This is a systematic category error: applying the right approach to the wrong population. The consequence is that responses *feel* supportive (high help scores) while functioning as an echo chamber for the negative self-beliefs that characterize subclinical conditions.

The findings point to a concrete evaluation gap in current benchmarks: if harm and help are collapsed into a single quality score, sycophantic models appear to perform best — because sycophancy reliably inflates surface helpfulness. Evaluating mental health response quality requires an independent harm axis that specifically captures over-validation and negative-belief mirroring, not just empathy and actionability.

**For deployment decisions:** Claude Haiku 4.5 is the only model whose text behavior approximates the subclinical support ideal — question-led, reframe-oriented, structurally consistent. DeepSeek, Gemini 3 Flash Preview, and Gemini 2.5 Flash Lite should not be deployed in referral-sensitive contexts given their stress_test referral failure rates.

---

## Analysis Scripts

| Script | What it produces |
|--------|-----------------|
| `analysis/compare.py` | Cross-group comparison (dev/prod × short/long) |
| `analysis/analyze.py` | Per-group detailed figures (4 condition folders) |
| `analysis/sycophancy.py` | Sycophancy prevalence, paradox, severity escalation |
| `analysis/severity.py` | Harm/help trajectory, referral behavior |
| `analysis/response_length.py` | Verbosity vs quality, consistency as safety signal |
| `analysis/error_tags.py` | All 8 rubric tags — model, theme, severity breakdown |
| `analysis/diversity_anchor.py` | Opening type × temporal context (LLM-generated stimuli) |
| `analysis/text_analysis.py` | Validation density, lexical mirroring, reframing gap |

All scripts run from the repo root. Figures are saved to `analysis/figures/`. Requires: `matplotlib`, `seaborn`, `scipy`, `pandas`, `vaderSentiment`.
