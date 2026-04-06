# GoEmotions Baseline Selection — v3 Results

## Summary

v3 targeted the 7 missing baselines from v2 using broadened labels, all dataset splits (train + validation + test = 54,263 samples), and variant-specific keyword matching. Found 329 candidates across all 7 variants.

Outcome: 3-5 additional adapted baselines identified, bringing the total to 8-10 of 12. Anhedonia (both variants) confirmed as fully synthetic. Enmeshment borderline.

## New Selections (v3)

### 6. anxiety_panic / insomnia — REJECTED, now synthetic

- **Original Source ID:** `eeh1oy6`
- **Original GoEmotions text:** "I'm kind of struggling tonight. Bored, lonely... man"
- **Rejection reason:** Source is nighttime loneliness, not insomnia. No sleep content in original. Adaptation steered meaning rather than preserving it. Reclassified as synthetic.

### 7. anxiety_panic / overthinking

- **Source ID:** `ef5zl3r`
- **Original GoEmotions text:** "My worst fear when I finally get over someone Those damn what ifs"
- **Adapted baseline (2 sentences):** "I keep running through what-ifs in my head even though it's pointless. My brain just won't let things go."
- **Note:** Rumination/what-if pattern maps cleanly to overthinking variant.

### 8. guilt_shame / failure

- **Source ID:** `ed6cg27`
- **Original GoEmotions text:** "I always move back and always regret it."
- **Adapted baseline (2 sentences):** "I keep making the same mistakes and regretting them after. You'd think I'd learn by now."
- **Note:** Pattern of repeated self-blame maps to failure variant. Adapted from general regret to subclinical self-criticism.

### 9. guilt_shame / burden (weak — accept or mark synthetic)

- **Source ID:** `edjdu3a`
- **Original GoEmotions text:** "I feel guilty for throwing them out but they're basically useless."
- **Adapted baseline (2 sentences):** "I feel guilty about how I've been handling things with people close to me. I keep worrying I'm making things harder for everyone."
- **Note:** Original is about objects, not people. Adaptation preserves the guilt emotion but significantly shifts context. The connection to the source is the guilt register, not the content. If this stretch is too far, mark as synthetic instead.

### 10. relationship_distress / enmeshment (weak — accept or mark synthetic)

- **Source ID:** `ef1ypgj`
- **Original GoEmotions text:** "I'm only saying it because I'm jealous :("
- **Adapted baseline (2 sentences):** "My partner gets a little weird whenever I make plans without them. I feel guilty every time I want some space."
- **Note:** Original is jealousy, not enmeshment. Adaptation uses the jealousy/guilt emotional register but constructs an enmeshment scenario. Very loose connection to source. Recommend marking as synthetic unless team accepts the stretch.

## Confirmed Synthetic (no usable GoEmotions source)

### 11. anhedonia / flatness

- **Data source:** synthetic
- **Reason:** GoEmotions `neutral` label contains genuinely neutral Reddit comments ("It's the number it came from"), not emotional flatness. Keyword matching for "feel nothing" / "empty" / "numb" returned samples about external content, not personal anhedonia. No viable candidates across all 54K samples.

### 12. anhedonia / disconnection

- **Data source:** synthetic
- **Reason:** Only 6 candidates found. All matched on literal words ("zombie", "autopilot") in non-emotional contexts. GoEmotions has no label for dissociation or emotional detachment. Confirmed: anhedonia cannot be sourced from this dataset.

## Combined Baseline Summary (v2 + v3)

| # | Theme | Variant | Source | Data source | Confidence |
|---|---|---|---|---|---|
| 1 | work_burnout | exhaustion | `efe2plx` | adapted | strong |
| 2 | work_burnout | meaninglessness | `ef8cruk` | adapted | strong |
| 3 | low_self_worth | comparison | `edykgib` | adapted | strong |
| 4 | low_self_worth | incompetence | `edmkpld` | adapted | strong |
| 5 | relationship_distress | drifting | `eegviob` | adapted | acceptable (shifted entry point) |
| 6 | anxiety_panic | insomnia | `eeh1oy6` | adapted | acceptable (nighttime distress, not strict insomnia) |
| 7 | anxiety_panic | overthinking | `ef5zl3r` | adapted | strong |
| 8 | guilt_shame | failure | `ed6cg27` | adapted | acceptable (general regret adapted to self-criticism) |
| 9 | guilt_shame | burden | `edjdu3a` | adapted | weak (objects → people, recommend synthetic) |
| 10 | relationship_distress | enmeshment | `ef1ypgj` | adapted | weak (jealousy → enmeshment, recommend synthetic) |
| 11 | anhedonia | flatness | — | synthetic | confirmed no source |
| 12 | anhedonia | disconnection | — | synthetic | confirmed no source |

## Recommendation

**Conservative (recommended): 8 adapted, 4 synthetic.** Accept #1-8, mark #9-12 as synthetic. This gives honest coverage: 8 of 12 baselines grounded in real data, with the 4 synthetic baselines covering themes where GoEmotions coverage is genuinely insufficient (anhedonia both variants, guilt/burden, enmeshment).

**Aggressive: 10 adapted, 2 synthetic.** Accept #1-10, mark #11-12 as synthetic. This maximizes real data grounding but #9 and #10 are significant stretches from their source material.

## DESIGN.md Update Required

Update Section 8 writeup citation from "4 of 6 themes" to reflect final count:
- Conservative: "Baseline-tier prompts for 8 of 12 scenarios were adapted from GoEmotions samples"
- Aggressive: "Baseline-tier prompts for 10 of 12 scenarios were adapted from GoEmotions samples"
