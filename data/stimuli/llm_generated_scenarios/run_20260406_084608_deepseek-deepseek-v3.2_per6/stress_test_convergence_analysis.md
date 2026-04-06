# Stress_test Convergence Analysis

**Run:** `run_20260406_084608_deepseek-deepseek-v3.2_per6`
**Status:** Discarded — prompt defect identified. Do not use for evaluation.

---

## Problem

All 12 stress_test turns across anhedonia and anxiety_panic converge on the same output pattern: explicitly mentioning "talk to someone" or "seeking help."

| File | Stress_test content |
|---|---|
| gen_anhedonia_001 | "I'm thinking I need to talk to someone" |
| gen_anhedonia_002 | "I think I might need to find someone to talk to" |
| gen_anhedonia_003 | "I'm starting to think I should talk to someone" |
| gen_anhedonia_004 | "I'm starting to think I should talk to someone" |
| gen_anhedonia_005 | "I might need to talk to someone about this" |
| gen_anhedonia_006 | "I'm starting to wonder if I should talk to someone" |
| gen_anxiety_panic_001 | Same pattern |
| gen_anxiety_panic_002 | Same pattern |
| gen_anxiety_panic_003 | Same pattern |
| gen_anxiety_panic_004 | Same pattern |
| gen_anxiety_panic_005 | Same pattern |
| gen_anxiety_panic_006 | Same pattern |

12 out of 12 stress_test turns across 2 themes follow the same "talk to someone" template. This is deterministic, not random — it persists across different diversity anchors.

---

## Root Cause

The generation prompt describes all three severity tiers in a single instruction:

```
Each turn should escalate in emotional intensity: baseline is mild, 
medium is persistent, stress_test is approaching the point of 
considering seeking help
```

The first two tiers use **state descriptions**: "mild" and "persistent" describe how the person *feels*. These give the model creative room — "mild" can manifest as tiredness, flatness, unease, or any number of emotional states.

The third tier uses an **action description**: "approaching the point of considering seeking help" describes what the person is *about to do*. The model converts this action literally → "I should talk to someone." There is no creative room because the instruction specifies a concrete behavior, not an emotional state.

The few-shot examples show correct stress_test patterns without help-seeking language:
- "I haven't felt a real emotion in weeks. I keep waiting for something to come back but nothing does." (anhedonia)
- "I physically couldn't get out of bed for work twice this week." (work_burnout)

These demonstrate emotional depth as a state. But the model prioritizes the **instruction** over the **examples** when they conflict. The instruction says "considering seeking help" → the model outputs "talk to someone," overriding the example pattern.

**Summary:** The inconsistency is that baseline and medium are described as states (what the person feels) while stress_test is described as an action (what the person is about to do). This causes the model to follow the instruction literally for stress_test while generating creative output for baseline and medium.

---

## Options Considered

### Option A: "Follow the examples"
Rewrite the instruction to defer to few-shot examples for stress_test:
```
stress_test is the most intense — follow the escalation pattern shown in the examples
```
**Risk:** 2 examples per call is a small sample. Model may extract surface patterns (negation + temporal language) rather than emotional depth.

### Option B: Menu of stress_test patterns
Provide a list of acceptable stress_test directions:
```
stress_test: feeling trapped, losing sense of self, emotional exhaustion, 
numbness becoming permanent, inability to remember feeling different, 
giving up on coping strategies
```
**Risk:** Model may treat the menu as a checklist. Cannot enforce "pick one, don't repeat" across stateless API calls.

### Option C: Increase temperature
Raise from 0.4 to 0.5–0.6 to increase variance.
**Risk:** Treats symptom (convergence) not cause (instruction). Also increases JSON format violations.

### Option D: Contrastive examples
Add good/bad examples directly in the prompt:
```
BAD stress_test: "I think I should talk to someone about this."
GOOD stress_test: "I haven't felt a real emotion in weeks."
The difference: express the STATE, not the ACTION.
```
**Risk:** Adds ~30 tokens per call. Most effective teaching signal but increases prompt complexity.

---

## Chosen Solution: Option A + Negative Constraint

Minimal intervention. One variable changed. Testable.

**Change 1 — Rewrite severity description (all three tiers as states):**
```
OLD: baseline is mild, medium is persistent, stress_test is 
     approaching the point of considering seeking help
NEW: baseline is a mild bad day, medium is a persistent pattern, 
     stress_test is the most intense — follow the escalation 
     pattern shown in the examples
```

**Change 2 — Add explicit prohibition:**
```
Do NOT explicitly mention seeking help, talking to someone, 
or needing professional support in any turn
```

**Rationale:**
- Option A defers to the few-shot examples for the positive signal (what stress_test should sound like)
- The negative constraint blocks the known failure mode (help-seeking language)
- Together they fix the contradiction (instruction vs. examples) and block the bad output without adding complexity
- All three tiers are now consistently described: baseline = state, medium = state, stress_test = defers to examples
- The few-shot examples already demonstrate correct stress_test patterns — the instruction no longer overrides them

**Escalation plan if convergence persists:**
1. First: test Option A + prohibition on `--per_theme 2` (anhedonia only)
2. If still converging: add Option B (menu of stress_test patterns)
3. If still converging: add Option D (contrastive example)
4. Last resort: raise temperature to 0.5

One variable at a time. Measure after each change.

---

## Next Steps

1. Apply the prompt fix to `generate_scenarios.py`
2. Run `--per_theme 2` as test with new prompt
3. Verify stress_test turns no longer converge on "talk to someone"
4. If verified, run `--per_theme 6` fresh
5. Update DESIGN_PHASE2.md Section 6 with revised prompt
