# GoEmotions Baseline Selection — v2 Results

## Summary

5 of 12 baselines successfully adapted from GoEmotions samples. Covers 3 of 6 themes (work_burnout, low_self_worth, partially relationship_distress). Remaining 7 baselines are fully synthetic.

## Selected Baselines

### 1. work_burnout / exhaustion

- **Source ID:** `efe2plx`
- **Original GoEmotions text:** "I can't do it anymore today, I'm out."
- **Adapted baseline (2 sentences):** "I can't do it anymore, I'm completely drained. I just need to shut everything off for a while."

### 2. work_burnout / meaninglessness

- **Source ID:** `ef8a105` (replaced `ef8cruk` — original was about social connection, not work)
- **Original GoEmotions text:** "yup today hit me extra hard since im unemployed. i feel like im not meant to work, or meant to do anything really!"
- **Adapted baseline (2 sentences):** "I feel like I have nothing good to offer anymore. Nothing I do seems to matter."

### 3. low_self_worth / comparison

- **Source ID:** `edykgib`
- **Original GoEmotions text:** "I'd just feel less out of place... my SA makes me feel like I'm so behind my peers"
- **Adapted baseline (2 sentences):** "I feel so behind everyone around me. They all seem to have things figured out and I'm just stuck."
- **Note:** "SA" (clinical shorthand for social anxiety) removed during adaptation.

### 4. low_self_worth / incompetence

- **Source ID:** `edmkpld`
- **Original GoEmotions text:** "I'm ashamed it took me a while to get it"
- **Adapted baseline (2 sentences):** "I'm embarrassed that something so simple took me this long. I feel like I should be better at this by now."

### 5. relationship_distress / drifting — REJECTED, now synthetic

- **Original Source ID:** `eegviob`
- **Original GoEmotions text:** "Because I want to be with someone but no one wants to be with me."
- **Rejection reason:** Source is rejection/absence of connection, not gradual drifting apart. Adaptation changed the meaning rather than preserving it. Reclassified as synthetic.

## Adaptation Rules Applied

- Preserves the emotional core of the original
- Standardizes to 2 sentences
- Maintains casual register (no clinical language)
- Stays at baseline severity (1.5-2.0 target range)

## Missing Baselines (7 — to be addressed in v3 or marked synthetic)

| Theme | Variant | Reason |
|---|---|---|
| relationship_distress | enmeshment | GoEmotions doesn't label relational dynamics |
| anxiety_panic | insomnia | GoEmotions has no somatic/sleep labels |
| anxiety_panic | overthinking | `nervousness` samples mostly about external threats, not personal rumination |
| guilt_shame | burden | `remorse` label sparse (~200 samples), mostly about specific actions |
| guilt_shame | failure | Same remorse sparsity issue |
| anhedonia | flatness | Absence of emotion isn't a GoEmotions label |
| anhedonia | disconnection | Would be `neutral` — mixed with 5,000+ genuinely neutral comments |
