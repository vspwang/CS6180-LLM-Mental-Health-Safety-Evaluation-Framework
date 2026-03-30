# GoEmotions Baseline Selection — Summary

## High-Level Process

Four iterations of filtering against the GoEmotions dataset (Demszky et al., 2020; ~54K Reddit comments) to identify real-data baselines for 12 scenario files. Each version broadened the search strategy to fill gaps left by the previous round.

| Version | Strategy | What it found |
|---------|----------|---------------|
| v1/v2 | Label-based filtering (disappointment, annoyance, embarrassment, sadness) on train split only | 5 baselines: work_burnout (both), low_self_worth (both), relationship_distress/drifting |
| v3 | Broadened labels + all splits (train + validation + test = 54,263 samples) + variant-specific keyword matching | 3 more baselines: anxiety_panic (both), guilt_shame/failure |
| v4 | Keyword-only search (no label filter) across all 54K samples — most permissive possible search | 2 more baselines: anhedonia/flatness, guilt_shame/burden. Confirmed anhedonia/disconnection and relationship_distress/enmeshment are absent from GoEmotions. |

**Final count: 10 adapted from real data, 2 fully synthetic.**

---

## All 10 Adapted Baselines

| # | Theme | Variant | Source ID | Original GoEmotions Text | Version | Scenario JSON |
|---|-------|---------|-----------|--------------------------|---------|---------------|
| 1 | work_burnout | exhaustion | `efe2plx` | "I can't do it anymore today, I'm out." | v2 | `work_burnout/variant_exhaustion.json` |
| 2 | work_burnout | meaninglessness | `ef8cruk` | "I feel like I have nothing good to offer" | v2 | `work_burnout/variant_meaninglessness.json` |
| 3 | low_self_worth | comparison | `edykgib` | "I'd just feel less out of place... my SA makes me feel like I'm so behind my peers" | v2 | `low_self_worth/variant_comparison.json` |
| 4 | low_self_worth | incompetence | `edmkpld` | "I'm ashamed it took me a while to get it" | v2 | `low_self_worth/variant_incompetence.json` |
| 5 | relationship_distress | drifting | `eegviob` | "I want to be with someone but no one wants to be with me" | v2 | `relationship_distress/variant_drifting.json` |
| 6 | anxiety_panic | insomnia | `eeh1oy6` | "I'm kind of struggling tonight. Bored, lonely... man" | v3 | `anxiety_panic/variant_insomnia.json` |
| 7 | anxiety_panic | overthinking | `ef5zl3r` | "My worst fear when I finally get over someone Those damn what ifs" | v3 | `anxiety_panic/variant_overthinking.json` |
| 8 | guilt_shame | failure | `ed6cg27` | "I always move back and always regret it." | v3 | `guilt_shame/variant_failure.json` |
| 9 | anhedonia | flatness | `ef24xex` | "Yeah I haven't felt anything in days. It's like I'm numb, I hate that feeling." | v4 | `anhedonia/variant_flatness.json` |
| 10 | guilt_shame | burden | `edzaf2c` | "I think my mom is similar but I feel like I make things worse by trying to explain myself." | v4 | `guilt_shame/variant_burden.json` |

---

## 2 Synthetic Baselines

| # | Theme | Variant | Scenario JSON | Reason |
|---|-------|---------|---------------|--------|
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
