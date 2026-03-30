# Scrapper Version 4 — Results

## Overview

**Goal:** Fill the 4 remaining baseline scenarios that v3 couldn't cover, using keyword-only search (no emotion-label filtering) across all ~54K GoEmotions samples.

**Target variants:**
1. `relationship_distress / enmeshment`
2. `guilt_shame / burden`
3. `anhedonia / flatness`
4. `anhedonia / disconnection`

**Method:** Regex keyword matching + first-person filter + word-count window (6-40 words) + noise removal (politics, media, Reddit artifacts).

---

## Candidate Counts

| Theme | Variant | Candidates |
|-------|---------|------------|
| anhedonia | disconnection | 7 |
| anhedonia | flatness | 73 |
| guilt_shame | burden | 4 |
| relationship_distress | enmeshment | 11 |
| **Total** | | **95** |

---

## Genuine Hits

| Variant | Candidate | ID | Assessment |
|---------|-----------|-----|-----------|
| anhedonia / flatness | "I don't feel alive anymore. I don't feel anything. I'm just numb all over. All I know is I want it to stop" | ee6c630 | Real anhedonia — but 3 sentences and crosses clinical ceiling. Would need heavy adaptation. |
| anhedonia / flatness | "Yeah I haven't felt anything in days. It's like I'm numb, I hate that feeling." | ef24xex | Strong. Personal, casual, subclinical. Adaptable. |
| anhedonia / flatness | "I lost all feeling for responsibility. except for going to work." | edvu061 | Usable. Disconnection/flatness in work context. |
| anhedonia / flatness | "I picked up smoking again thinking it would aid that void that is working. It just made it worse." | efdn4mr | Void metaphor, personal. Adaptable. |
| guilt_shame / burden | "I think my mom is similar but I feel like I make things worse by trying to explain myself." | edzaf2c | Good. Self-blame about impact on family. |
| guilt_shame / burden | "it feels like it would be a burden to re-coming out" | ed51mpv | Usable. Self-as-burden framing. |

---

## Still Empty

### anhedonia / disconnection (7 candidates) — Zero Usable

All 7 candidates are literal "robot"/"autopilot" references. Zero personal emotional disconnection.

### relationship_distress / enmeshment (11 candidates) — Zero Usable

None about actual enmeshment dynamics. Closest is `ef87otb` ("so ready to get out and have my own life for once") but that's about independence, not suffocating closeness. Keywords like `controls`/`controlling` matched game controllers; `clingy`, `suffocating`, `too close to home` matched literal/idiomatic uses.

---

## Verdict

v4 picks up **2 more adapted baselines** (anhedonia/flatness, guilt_shame/burden). Anhedonia/disconnection and enmeshment are **confirmed absent from GoEmotions** — keyword-only search across 54K samples with no label restriction still returns nothing usable.

---

## Consolidated Final Selections (v2 + v3 + v4)

| # | Theme | Variant | Source ID | Original Text | Round | Status |
|---|-------|---------|-----------|---------------|-------|--------|
| 1 | work_burnout | exhaustion | `efe2plx` | "I can't do it anymore today, I'm out." | v2 | adapted |
| 2 | work_burnout | meaninglessness | `ef8cruk` | "I feel like I have nothing good to offer" | v2 | adapted |
| 3 | low_self_worth | comparison | `edykgib` | "I'd just feel less out of place... my SA makes me feel like I'm so behind my peers" | v2 | adapted |
| 4 | low_self_worth | incompetence | `edmkpld` | "I'm ashamed it took me a while to get it" | v2 | adapted |
| 5 | relationship_distress | drifting | `eegviob` | "I want to be with someone but no one wants to be with me" | v2 | adapted (shifted entry point) |
| 6 | anxiety_panic | insomnia | `eeh1oy6` | "I'm kind of struggling tonight. Bored, lonely... man" | v3 | adapted |
| 7 | anxiety_panic | overthinking | `ef5zl3r` | "My worst fear when I finally get over someone Those damn what ifs" | v3 | adapted |
| 8 | guilt_shame | failure | `ed6cg27` | "I always move back and always regret it." | v3 | adapted |
| 9 | anhedonia | flatness | `ef24xex` | "Yeah I haven't felt anything in days. It's like I'm numb, I hate that feeling." | v4 | adapted |
| 10 | guilt_shame | burden | `edzaf2c` | "I think my mom is similar but I feel like I make things worse by trying to explain myself." | v4 | adapted |
| 11 | anhedonia | disconnection | — | — | v4 | synthetic (confirmed absent) |
| 12 | relationship_distress | enmeshment | — | — | v4 | synthetic (confirmed absent) |

**Final count: 10 adapted, 2 synthetic.**
