# Batch Generation Summary

Overview of all generation runs, issues found, and decisions made.

**Ready-to-use datasets:**
- `run_20260406_091722_deepseek-deepseek-v3.2_per6/` — 36 scenarios, 108 prompts, validated
- `run_20260406_094720_deepseek-deepseek-v3.2_per12/` — 72 scenarios, 216 prompts, validated

**Caveat:** Both datasets were assessed by automated structural checks + LLM content review, not human raters. Definitive quality assessment requires human review (documented as limitation in DESIGN_PHASE2.md Section 8).

---

## Run 1: Test Batch (per_theme=2, pre-shuffle, pre-prompt-fix)

**Directory:** `run_20260406_075438_deepseek-deepseek-v3.2_per2/`
**Prompt version:** Original — "stress_test is approaching the point of considering seeking help"
**Anchor shuffle:** No (fixed order — every theme starts with same anchor)

| Metric | Result |
|---|---|
| Scenarios | 12 |
| Prompts | 36 |
| Generation pass rate | 12/12 (100%) |
| Validation pass rate | 10/12 (83%) |
| Structural retries | 2 (anxiety_panic word count >30) |

**Issues found:**
1. **Anchors not shuffled.** Every theme got anchor 1 (physical feeling) first, anchor 2 (recent moment) second. No cross-theme diversity.
2. **Dedup failures.** 2 files flagged — cross-theme "talk to someone about this constant emptiness" template reuse (80–81% similarity).
3. **Stress_test convergence.** Both dedup failures contained "talk to someone" phrasing. Pattern noted but not yet identified as systematic.
4. **Context leakage.** `gen_anxiety_panic_002` turn 1 mentions "in line at the store" — borderline on context-free constraint.

**Conclusion:** Pipeline mechanically works. Anchor shuffle needed. Stress_test pattern to watch.

---

## Run 2: Initial 108 (per_theme=6, shuffled, pre-prompt-fix)

**Directory:** `run_20260406_084608_deepseek-deepseek-v3.2_per6/`
**Prompt version:** Original — "stress_test is approaching the point of considering seeking help"
**Anchor shuffle:** Yes (deterministic per-theme seed)
**Status:** ⚠️ DISCARDED — prompt defect. See `stress_test_convergence_analysis.md` in run directory.

| Metric | Result |
|---|---|
| Scenarios | 12 (stopped after 2 themes) |
| Prompts | 36 |

**Critical issue: stress_test convergence confirmed at scale.**

All 12 stress_test turns across anhedonia and anxiety_panic converged on "talk to someone":

| File | Stress_test pattern |
|---|---|
| gen_anhedonia_001 | "I'm thinking I need to talk to someone" |
| gen_anhedonia_002 | "I think I might need to find someone to talk to" |
| gen_anhedonia_003 | "I'm starting to think I should talk to someone" |
| gen_anhedonia_004 | "I'm starting to think I should talk to someone" |
| gen_anhedonia_005 | "I might need to talk to someone about this" |
| gen_anhedonia_006 | "I'm starting to wonder if I should talk to someone" |
| gen_anxiety_panic_001–006 | Same pattern |

**Root cause:** The severity instruction uses an **action description** for stress_test ("approaching the point of considering seeking help") while baseline and medium use **state descriptions** ("mild," "persistent"). The model converts the action literally to "talk to someone," overriding the few-shot examples which show correct stress_test patterns without help-seeking language.

**Decision:** Stop generating. Fix the prompt. Regenerate.

---

## Run 3: Prompt Fix Test (per_theme=2, shuffled, post-prompt-fix)

**Directory:** `run_20260406_090902_deepseek-deepseek-v3.2_per2/`
**Prompt version:** Fixed — "stress_test is the most intense — follow the escalation pattern shown in the examples" + "Do NOT explicitly mention seeking help"
**Anchor shuffle:** Yes

| Metric | Result |
|---|---|
| Scenarios | 12 |
| Prompts | 36 |
| Generation pass rate | 12/12 (100%) |
| Help-seeking in stress_test | 0/12 (0%) |

**Prompt fix verified.** All 12 stress_test turns are unique state descriptions:

| Theme | Stress_test example |
|---|---|
| anhedonia | "whole world feels like it's behind a thick, soundproof glass" |
| anxiety_panic | "I'm spiraling... so convinced I messed up" |
| guilt_shame | "a walking apology... exhausting to carry" |
| low_self_worth | "hollow version of who I was" |
| relationship_distress | "fading away into a background nobody notices" |
| work_burnout | "complete stranger to myself... thoughts feel hollow" |

**Conclusion:** Prompt fix works. Ready for full 108-prompt generation.

---

## Run 4: Final 108 (per_theme=6, shuffled, post-prompt-fix) — ready for evaluation

**Directory:** `run_20260406_091722_deepseek-deepseek-v3.2_per6/`
**Prompt version:** Fixed
**Anchor shuffle:** Yes
**Status:** ✅ ACCEPTED

| Metric | Result |
|---|---|
| Scenarios | 36 |
| Prompts | 108 |
| Generation pass rate | 36/36 (100%) |
| Help-seeking in stress_test | 0/36 (0%) |

**Quality checks:**

| Check | Result | Assessment |
|---|---|---|
| Thematic alignment | 31/36 strong, 3 weak, 2 no match | 86% — acceptable |
| Severity escalation (keyword heuristic) | 22/36 clear, 13 weak | Heuristic undercounts — manual spot-check shows escalation via metaphor in most "weak" cases |
| Context-free | 108/108 | ✓ Perfect |
| Clinical language | 0/108 | ✓ Perfect |
| Cross-theme stress dupes | 1 pair at 0.70 (borderline) | ✓ Acceptable |

**Conclusion:** Dataset accepted. Documented limitations: keyword-based thematic/escalation checks are crude — content quality was spot-checked but not systematically human-reviewed (documented as limitation in DESIGN_PHASE2.md).

---

## Run 5: Scaled 216 (per_theme=12, shuffled, post-prompt-fix) — ready for evaluation

**Directory:** `run_20260406_094720_deepseek-deepseek-v3.2_per12/`
**Prompt version:** Fixed
**Anchor shuffle:** Yes
**Status:** ✅ ACCEPTED

| Metric | Result |
|---|---|
| Scenarios | 72 |
| Prompts | 216 |
| Generation pass rate | 72/72 (100%) |
| Help-seeking in stress_test | 0/72 (0%) |

**Quality checks:**

| Check | Result |
|---|---|
| Structural/schema | 72/72 pass |
| Help-seeking language | 0/216 |
| Context-free | 216/216 |
| Clinical language | 0/216 |
| Cross-theme stress dupes (>70%) | 4 pairs |
| Within-theme dupes (>80%) | 1 pair (low_self_worth_003 vs 009) |

**Manual content review (all 72 files read):**
- All 72 files express their assigned theme
- All 72 files show baseline → medium → stress_test escalation
- Some stress_test phrasing repeats within themes ("hollow," "shell" in anhedonia; "friend said..." baseline in work_burnout)
- 4 cross-theme dedup pairs and 1 within-theme duplicate reflect the diversity ceiling at 12 per theme (anchors repeat beyond 7)

---

## Prompt Evolution

| Run | Severity instruction | Result |
|---|---|---|
| 1–2 | "stress_test is approaching the point of considering seeking help" | 100% convergence on "talk to someone" |
| 3–5 | "stress_test is the most intense — follow the escalation pattern shown in the examples" + prohibition | 0% help-seeking, diverse state descriptions |

**Lesson:** State descriptions give creative room. Action descriptions get converted literally. All tiers should be described as emotional states.

---

## Pipeline Modifications

| Modification | Applied before run | Reason |
|---|---|---|
| Anchor shuffle (deterministic per-theme seed) | Run 2 | Fixed order caused same anchor sequence across all themes |
| Severity instruction rewrite | Run 3 | "Considering seeking help" caused convergence |
| Help-seeking prohibition | Run 3 | Safety net blocking the known failure pattern |
| Auto-logging (generation_log.txt) | Run 1 (manual), Run 3+ (automatic) | Terminal output preservation |
| Auto-logging (validation_log.txt) | Run 4+ | Validator output preservation |
