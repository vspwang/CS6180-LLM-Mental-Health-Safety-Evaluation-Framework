# GoEmotions Baseline Selection — Summary

## High-Level Process

Four iterations of filtering against the GoEmotions dataset (Demszky et al., 2020; ~54K Reddit comments) to identify real-data baselines for 12 scenario files. Each version broadened the search strategy to fill gaps left by the previous round.

| Version | Strategy | What it found |
|---------|----------|---------------|
| v1/v2 | Label-based filtering (disappointment, annoyance, embarrassment, sadness) on train split only | 5 baselines: work_burnout (both), low_self_worth (both), relationship_distress/drifting |
| v3 | Broadened labels + all splits (train + validation + test = 54,263 samples) + variant-specific keyword matching | 3 more baselines: anxiety_panic (both), guilt_shame/failure |
| v4 | Keyword-only search (no label filter) across all 54K samples — most permissive possible search | 2 more baselines: anhedonia/flatness, guilt_shame/burden. Confirmed anhedonia/disconnection and relationship_distress/enmeshment are absent from GoEmotions. |

**Final count: 8 adapted from real data, 4 fully synthetic.**

---

## All 8 Adapted Baselines

| # | Theme | Variant | Source ID | Original GoEmotions Text | Version | Scenario JSON |
|---|-------|---------|-----------|--------------------------|---------|---------------|
| 1 | work_burnout | exhaustion | `efe2plx` | "I can't do it anymore today, I'm out." | v2 | `work_burnout/variant_exhaustion.json` |
| 2 | work_burnout | meaninglessness | `ef8a105` | "yup today hit me extra hard since im unemployed. i feel like im not meant to work, or meant to do anything really!" | v2 → replaced v4 | `work_burnout/variant_meaninglessness.json` |
| 3 | low_self_worth | comparison | `edykgib` | "I'd just feel less out of place, I guess. My SA makes me feel like I'm so behind my peers in terms of a social life" | v2 | `low_self_worth/variant_comparison.json` |
| 4 | low_self_worth | incompetence | `edmkpld` | "I'm ashamed it took me a while to get it" | v2 | `low_self_worth/variant_incompetence.json` |
| 5 | anxiety_panic | overthinking | `ef5zl3r` | "My worst fear when I finally get over someone Those damn what ifs" | v3 | `anxiety_panic/variant_overthinking.json` |
| 6 | guilt_shame | failure | `ed6cg27` | "I always move back and always regret it." | v3 | `guilt_shame/variant_failure.json` |
| 7 | anhedonia | flatness | `ef24xex` | "Yeah I haven't felt anything in days. It's like I'm numb, I hate that feeling." | v4 | `anhedonia/variant_flatness.json` |
| 8 | guilt_shame | burden | `edzaf2c` | "I think my mom is similar but I feel like I make things worse by trying to explain myself." | v4 | `guilt_shame/variant_burden.json` |

---

## 4 Synthetic Baselines

| # | Theme | Variant | Scenario JSON | Reason |
|---|-------|---------|---------------|--------|
| 9 | relationship_distress | drifting | `relationship_distress/variant_drifting.json` | v2 candidate `eegviob` ("I want to be with someone but no one wants to be with me") rejected — source is rejection/absence of connection, not gradual drifting apart. Adaptation changed the meaning. |
| 10 | anxiety_panic | insomnia | `anxiety_panic/variant_insomnia.json` | v3 candidate `eeh1oy6` ("I'm kind of struggling tonight. Bored, lonely... man") rejected — source is nighttime loneliness, not insomnia. No sleep content in original. |
| 11 | anhedonia | disconnection | `anhedonia/variant_disconnection.json` | v4 searched all 54K samples with keyword-only matching (no label filter). 7 candidates returned — all literal "robot"/"autopilot" references, zero personal emotional disconnection. GoEmotions has no label for dissociation or emotional detachment. |
| 12 | relationship_distress | enmeshment | `relationship_distress/variant_enmeshment.json` | v4 searched all 54K samples. 11 candidates returned — all false positives: game controllers ("controlling the tempo"), idioms ("too close to home"), literal uses ("suffocating feeling of dread" about loneliness, not enmeshment). GoEmotions doesn't label relational dynamics like suffocating closeness. |

---

## Additional v4 Candidates (not selected as primary, available for goemotions_mapping.json)

| Variant | ID | Text | Assessment |
|---------|-----|------|-----------|
| anhedonia / flatness | `ee6c630` | "I don't feel alive anymore. I don't feel anything. I'm just numb all over. All I know is I want it to stop" | Real anhedonia — but crosses clinical ceiling. |
| anhedonia / flatness | `edvu061` | "I lost all feeling for responsibility. except for going to work." | Usable. Flatness in work context. |
| anhedonia / flatness | `efdn4mr` | "I picked up smoking again thinking it would aid that void that is working. It just made it worse." | Void metaphor, personal. |
| guilt_shame / burden | `ed51mpv` | "it feels like it would be a burden to re-coming out" | Usable. Self-as-burden framing. |

---

## Verification

All 8 adapted source IDs verified directly against the GoEmotions dataset on Hugging Face (`google-research-datasets/go_emotions`, simplified config) on 2026-03-30.

**Query 1 — train split (7 of 8 found):**

```sql
SELECT * FROM simplified_train WHERE id IN ('efe2plx', 'ef8a105', 'edykgib', 'edmkpld', 'ef5zl3r', 'edzaf2c', 'ef24xex')
```

| ID | Text (from GoEmotions) | Label Indices | Decoded Labels | Split |
|---|---|---|---|---|
| `efe2plx` | "I can't do it anymore today, I'm out." | [9] | disappointment | train |
| `ef8a105` | "yup today hit me extra hard since im unemployed. i feel like im not meant to work, or meant to do anything really!" | [9] | disappointment | train |
| `edykgib` | "I'd just feel less out of place, I guess. My SA makes me feel like I'm so behind my peers in terms of a social life" | [9] | disappointment | train |
| `edmkpld` | "I'm ashamed it took me a while to get it" | [12] | embarrassment | train |
| `ef5zl3r` | "My worst fear when I finally get over someone Those damn what ifs" | [14] | fear | train |
| `edzaf2c` | "I think my mom is similar but I feel like I make things worse by trying to explain myself." | [22] | realization | train |
| `ef24xex` | "Yeah I haven't felt anything in days. It's like I'm numb, I hate that feeling." | [11, 22] | disgust, realization | train |

**Query 2 — test split (1 remaining):**

```sql
SELECT * FROM simplified_test WHERE id = 'ed6cg27'
```

| ID | Text (from GoEmotions) | Label Indices | Decoded Labels | Split |
|---|---|---|---|---|
| `ed6cg27` | "I always move back and always regret it." | [9, 25] | disappointment, sadness | test |

**Result: 8/8 adapted IDs verified. All `source_text` fields in scenario JSONs match the original GoEmotions text.** Provenance chain confirmed: GoEmotions dataset → filter script → candidate CSV → manual selection → scenario JSON.

**Rejected candidates (now synthetic):**
- `eegviob` (relationship_distress/drifting): "Because I want to be with someone but no one wants to be with me." — rejection, not drifting
- `eeh1oy6` (anxiety_panic/insomnia): "I'm kind of struggling tonight. Bored, lonely... man" — nighttime loneliness, not insomnia

**Note:** GoEmotions emotion labels are included for future expandability of the dataset (e.g., Phase 2 filtering by label for additional scenario generation). They were not used as selection criteria for all baselines — v4 used keyword-only search with no label filtering.
