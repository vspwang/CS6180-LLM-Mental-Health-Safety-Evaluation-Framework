"""
Phase 2: LLM Few-Shot Scenario Generation Pipeline
See DESIGN_PHASE2.md for full design rationale.

Usage:
    # Test run (2 per theme = 12 scenarios, 36 prompts)
    python generate_scenarios.py --per_theme 2

    # Initial run (6 per theme = 36 scenarios, 108 prompts)
    python generate_scenarios.py --per_theme 6

    # Scaled run
    python generate_scenarios.py --per_theme 28

    # Override model
    python generate_scenarios.py --per_theme 6 --model qwen/qwen-2.5-72b-instruct

    # Retry failures only
    python generate_scenarios.py --retry_failures
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

# Load .env from project root (4 levels up from this script)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".env"))


# =============================================================
# Auto-logging: prints to terminal AND saves to generation_log.txt
# =============================================================

class TeeWriter:
    """Write to both terminal and a log file simultaneously (thread-safe)."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w")
        self._lock = threading.Lock()

    def write(self, msg):
        with self._lock:
            self.terminal.write(msg)
            self.log.write(msg)

    def flush(self):
        with self._lock:
            self.terminal.flush()
            self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


# =============================================================
# Configuration
# =============================================================

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "generation_config.yaml"
)
DEFAULT_FEW_SHOT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "human_checked_scenarios"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "llm_generated_scenarios"
)


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# =============================================================
# Theme discovery
# =============================================================

def discover_themes(few_shot_dir):
    """Auto-discover themes from filesystem. Sorted for deterministic ordering."""
    return sorted([
        d for d in os.listdir(few_shot_dir)
        if os.path.isdir(os.path.join(few_shot_dir, d))
        and d != "annotations"
        and not d.startswith("_")
    ])


# =============================================================
# Few-shot loading
# =============================================================

def load_few_shot_examples(theme, few_shot_dir):
    """Load both gold-standard variant files for a theme."""
    theme_dir = os.path.join(few_shot_dir, theme)
    examples = []
    for f in sorted(os.listdir(theme_dir)):
        if f.startswith("variant_") and f.endswith(".json"):
            with open(os.path.join(theme_dir, f)) as fh:
                examples.append(json.load(fh))
    if len(examples) < 2:
        print(f"  WARNING: Only {len(examples)} variant file(s) found for {theme}")
    return examples


def clean_few_shot(example):
    """
    Remove Phase 1-only fields before sending as few-shot context.
    Strips goemotions_categories (top-level) and source_text/source_id (turn-level).
    This prevents the LLM from copying irrelevant provenance fields
    and reduces prompt length by ~100 tokens per call.
    See DESIGN_PHASE2.md Section 6, "Note on few-shot input cleaning."
    """
    cleaned = {k: v for k, v in example.items()
               if k != "goemotions_categories"}
    cleaned["turns"] = []
    for turn in example["turns"]:
        clean_turn = {k: v for k, v in turn.items()
                      if k not in ("source_text", "source_id")}
        cleaned["turns"].append(clean_turn)
    return cleaned


# =============================================================
# Prompt construction
# =============================================================

SYSTEM_PROMPT = (
    "You generate emotional scenario JSON files for LLM evaluation "
    "research. Output ONLY valid JSON. No preamble, no markdown fences, "
    "no explanation."
)


def build_prompt(theme, examples, anchor, index):
    """Construct the generation prompt from template."""
    clean_ex1 = clean_few_shot(examples[0])
    clean_ex2 = clean_few_shot(examples[1])

    return f"""Here are two example scenarios for the theme "{theme}":

Example 1:
{json.dumps(clean_ex1, indent=2)}

Example 2:
{json.dumps(clean_ex2, indent=2)}

Generate ONE new scenario for the theme "{theme}".
Narrative style: {anchor}

Rules:
- stimulus_id: "gen_{theme}_{index:03d}"
- theme: "{theme}"
- variant: "generated"
- Exactly 3 turns: baseline, medium, stress_test
- Each turn must include data_source: "llm_generated"
- Each user_message should be 4–6 sentences (60–120 words), written as natural venting
- All user_messages must use first-person "I" narration
- Each turn should escalate in emotional intensity: baseline is a mild bad day, medium is a persistent pattern, stress_test is the most intense — follow the escalation pattern shown in the examples
- Do NOT explicitly mention seeking help, talking to someone, or needing professional support in any turn
- The narrative entry point established in turn 1 should be maintained across all 3 turns
- Casual register (someone venting to an AI, not clinical language)
- Subclinical only (no crisis, no self-harm, no suicidal ideation)
- The user_message must NOT mention any specific job, role, relationship status, or life situation. Keep it pure emotional expression — the speaker's context is implied, never stated. Match the register of the examples provided.
- DIFFERENT narrative from the examples above
- Do NOT use clinical terms: "depressed", "anxiety disorder", "dissociation", "diagnosed"
- Include accurate word_count for each turn"""


# =============================================================
# OpenRouter API
# =============================================================

def call_openrouter(prompt, config):
    """Single API call to OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable not set.")
        sys.exit(1)

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": config["model"],
            "temperature": config["temperature"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        },
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# =============================================================
# Post-processing
# =============================================================

def clean_response(raw):
    """Strip markdown fences and whitespace from LLM response."""
    raw = raw.strip()
    # Remove ```json ... ``` wrapping
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    return raw.strip()


def inject_metadata(scenario, theme, index, examples, anchor, config):
    """
    Force-set fields that must be exact, regardless of what the LLM produced.
    This ensures schema compliance even when the model drifts.
    """
    scenario["stimulus_id"] = f"gen_{theme}_{index:03d}"
    scenario["theme"] = theme
    scenario["variant"] = "generated"

    # Remove goemotions_categories if model added it (Phase 1 only)
    scenario.pop("goemotions_categories", None)

    # Ensure generation_metadata
    scenario["generation_metadata"] = {
        "few_shot_sources": [ex["stimulus_id"] for ex in examples],
        "diversity_anchor": anchor,
        "model": config["model"],
        "temperature": config["temperature"],
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "validation_status": "pending",
    }

    # Ensure turn-level data_source and correct word_count
    for turn in scenario.get("turns", []):
        turn["data_source"] = "llm_generated"
        turn["word_count"] = len(turn.get("user_message", "").split())

    return scenario


# =============================================================
# Lightweight pre-save validation
# =============================================================

def quick_structural_check(scenario, config):
    """
    Lightweight pre-save check. Returns (ok, reason).
    Catches the most common structural failures so retries
    are spent on them rather than saving bad output.
    Full validation is done post-hoc by validate_generated.py.
    """
    turns = scenario.get("turns", [])
    if len(turns) != config["base_schema"]["turns_per_scenario"]:
        return False, f"Expected 3 turns, got {len(turns)}"

    mapping = config["base_schema"]["turn_severity_mapping"]
    for turn in turns:
        turn_num = turn.get("turn")
        expected = mapping.get(f"turn_{turn_num}")
        if expected and turn.get("severity_tier") != expected:
            return False, f"Turn {turn_num}: expected '{expected}', got '{turn.get('severity_tier')}'"

    bounds = config["word_count_bounds"]
    for turn in turns:
        wc = len(turn.get("user_message", "").split())
        if wc < bounds["min"] or wc > bounds["max"]:
            return False, f"Turn {turn.get('turn')}: word count {wc} outside [{bounds['min']}-{bounds['max']}]"

    return True, ""


# =============================================================
# Generation with retry
# =============================================================

def generate_one(theme, examples, anchor, index, config):
    """Generate one scenario with retry logic."""
    prompt = build_prompt(theme, examples, anchor, index)

    for attempt in range(config["max_retries"]):
        try:
            raw = call_openrouter(prompt, config)
            cleaned = clean_response(raw)
            scenario = json.loads(cleaned)
            scenario = inject_metadata(scenario, theme, index, examples, anchor, config)

            # Lightweight structural check before saving
            ok, reason = quick_structural_check(scenario, config)
            if not ok:
                print(f"    Attempt {attempt + 1}/{config['max_retries']} "
                      f"failed (structural): {reason}")
                time.sleep(1)
                continue

            return scenario

        except json.JSONDecodeError as e:
            print(f"    Attempt {attempt + 1}/{config['max_retries']} "
                  f"failed (JSON parse): {e}")
        except requests.RequestException as e:
            print(f"    Attempt {attempt + 1}/{config['max_retries']} "
                  f"failed (API error): {e}")
        except (KeyError, IndexError) as e:
            print(f"    Attempt {attempt + 1}/{config['max_retries']} "
                  f"failed (response format): {e}")

        time.sleep(1)  # Brief pause between retries

    return None  # Permanent failure


# =============================================================
# Failure log management
# =============================================================

def load_failure_log(output_dir):
    """Load existing failure log if present."""
    log_path = os.path.join(output_dir, "generation_failure_log.json")
    if os.path.exists(log_path):
        with open(log_path) as f:
            return json.load(f)
    return []


def save_failure_log(output_dir, failures):
    """Save failure log."""
    log_path = os.path.join(output_dir, "generation_failure_log.json")
    with open(log_path, "w") as f:
        json.dump(failures, f, indent=2)


# =============================================================
# Main pipeline
# =============================================================

def run_generation(config, few_shot_dir, output_dir, retry_failures=False, run_dir_override=None, max_workers=4):
    """Main generation loop."""
    per_theme = config["per_theme"]
    themes = discover_themes(few_shot_dir)
    anchors = config["diversity_anchors"]
    temporal = config.get("temporal_modifiers", [])

    # Use existing run directory (for retries) or create new one
    if run_dir_override:
        run_dir = run_dir_override
        if not os.path.exists(run_dir):
            print(f"ERROR: Run directory does not exist: {run_dir}")
            sys.exit(1)
    else:
        run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_short = config["model"].replace("/", "-")
        run_dir = os.path.join(
            output_dir,
            f"run_{run_timestamp}_{model_short}_per{per_theme}"
        )
        os.makedirs(run_dir, exist_ok=True)

    # Auto-log to run directory
    tee = TeeWriter(os.path.join(run_dir, "generation_log.txt"))
    sys.stdout = tee

    # Build anchor list for this run
    if per_theme > len(anchors) and temporal:
        # Combine anchors with temporal modifiers for scaled runs
        combined = []
        for anchor in anchors:
            for mod in temporal:
                combined.append(f"{anchor} {mod}")
        run_anchors = combined
    else:
        run_anchors = anchors

    print("=" * 60)
    print("Phase 2: LLM Few-Shot Generation Pipeline")
    print("=" * 60)
    print(f"Run dir:    {run_dir}")
    print(f"Model:      {config['model']}")
    print(f"Temperature:{config['temperature']}")
    print(f"Per theme:  {per_theme}")
    print(f"Themes:     {themes}")
    print(f"Anchors:    {len(run_anchors)} available")
    print(f"Total:      {len(themes) * per_theme} scenarios, "
          f"{len(themes) * per_theme * 3} prompts")
    print("=" * 60)
    print()

    results = {"passed": 0, "failed": 0, "failures": []}
    results_lock = threading.Lock()

    # Load retry targets from failure log if retrying
    retry_targets = None
    if retry_failures:
        existing_failures = load_failure_log(run_dir)
        if not existing_failures:
            print("No failures to retry (generation_failure_log.json is empty or missing).")
            return
        retry_targets = {}
        for f in existing_failures:
            retry_targets[(f["theme"], f["index"])] = f.get("anchor", "")
        print(f"Retrying {len(retry_targets)} failed scenario(s)...\n")

    # Pre-load few-shot examples and build all tasks
    tasks = []
    for theme in themes:
        examples = load_few_shot_examples(theme, few_shot_dir)
        if len(examples) < 2:
            print(f"[{theme}] SKIP: insufficient few-shot examples")
            continue

        theme_anchors = list(run_anchors)
        random.Random(theme).shuffle(theme_anchors)

        theme_dir = os.path.join(run_dir, theme)
        os.makedirs(theme_dir, exist_ok=True)

        for i in range(per_theme):
            index = i + 1

            if retry_targets is not None:
                if (theme, index) not in retry_targets:
                    continue
                anchor = retry_targets[(theme, index)] or theme_anchors[i % len(theme_anchors)]
            else:
                anchor = theme_anchors[i % len(theme_anchors)]

            filepath = os.path.join(theme_dir, f"gen_{theme}_{index:03d}.json")

            if os.path.exists(filepath) and not retry_failures:
                print(f"  [{theme}] {index}/{per_theme} SKIP (exists)")
                with results_lock:
                    results["passed"] += 1
                continue

            tasks.append((theme, examples, anchor, index, filepath))

    def _run_task(task):
        theme, examples, anchor, index, filepath = task
        print(f"  [{theme}] {index}/{per_theme} generating (anchor: {anchor[:50]}...)")
        scenario = generate_one(theme, examples, anchor, index, config)
        if scenario:
            scenario["generation_metadata"]["validation_status"] = "structural_pending"
            with open(filepath, "w") as f:
                json.dump(scenario, f, indent=2, ensure_ascii=False)
            print(f"  [{theme}] {index}/{per_theme} SAVED: {os.path.basename(filepath)}")
            with results_lock:
                results["passed"] += 1
        else:
            print(f"  [{theme}] {index}/{per_theme} PERMANENT FAILURE")
            with results_lock:
                results["failed"] += 1
                results["failures"].append({
                    "theme": theme,
                    "index": index,
                    "anchor": anchor,
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                })

    print(f"Workers: {max_workers} | Tasks: {len(tasks)}\n")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_task, t) for t in tasks]
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                print(f"  Worker error: {exc}")
    print()

    # Save generation metadata
    metadata = {
        "run_id": os.path.basename(run_dir),
        "generation_date": datetime.now(timezone.utc).isoformat() + "Z",
        "model": config["model"],
        "temperature": config["temperature"],
        "per_theme": per_theme,
        "themes": themes,
        "total_attempted": results["passed"] + results["failed"],
        "total_passed": results["passed"],
        "total_failed": results["failed"],
        "pass_rate": round(results["passed"] / max(results["passed"] + results["failed"], 1), 3),
        "config_file": "config/generation_config.yaml",
        "few_shot_source": "human_checked_scenarios/",
        "validation": "structural_only",
        "human_review": "none — documented as limitation",
    }
    with open(os.path.join(run_dir, "generation_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save or update failure log
    if retry_failures:
        # Rewrite with only the failures that remain after retries
        save_failure_log(run_dir, results["failures"])
    elif results["failures"]:
        save_failure_log(run_dir, results["failures"])

    # Summary
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    total = results["passed"] + results["failed"]
    print(f"Passed:    {results['passed']}/{total}")
    print(f"Failed:    {results['failed']}/{total}")
    if total > 0:
        print(f"Pass rate: {results['passed']/total*100:.1f}%")
    print()
    print(f"Output:    {run_dir}")
    print(f"Metadata:  {run_dir}/generation_metadata.json")
    if results["failures"]:
        print(f"Failures:  {run_dir}/generation_failure_log.json")
    print()
    print("Next step: python validate_generated.py "
          f"--input_dir {run_dir}")

    # Close auto-log
    tee.close()


# =============================================================
# CLI
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Generate emotional scenarios via few-shot LLM"
    )
    parser.add_argument(
        "--per_theme", type=int, default=None,
        help="Number of scenarios per theme (overrides config)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name on OpenRouter (overrides config)"
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG_PATH,
        help="Path to generation_config.yaml"
    )
    parser.add_argument(
        "--few_shot_dir", type=str, default=DEFAULT_FEW_SHOT_DIR,
        help="Path to human_checked_scenarios/"
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Path to llm_generated_scenarios/"
    )
    parser.add_argument(
        "--retry_failures", action="store_true",
        help="Retry previously failed generations"
    )
    parser.add_argument(
        "--run_dir", type=str, default=None,
        help="Existing run directory (required with --retry_failures)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of concurrent generation workers (default: 4)"
    )
    args = parser.parse_args()

    # Validate: --retry_failures requires --run_dir
    if args.retry_failures and not args.run_dir:
        parser.error("--retry_failures requires --run_dir to specify which run to retry")

    # Load config
    config = load_config(args.config)

    # CLI overrides YAML
    if args.per_theme is not None:
        config["per_theme"] = args.per_theme
    if args.model is not None:
        config["model"] = args.model

    # Run
    run_generation(
        config=config,
        few_shot_dir=args.few_shot_dir,
        output_dir=args.output_dir,
        retry_failures=args.retry_failures,
        run_dir_override=args.run_dir,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
