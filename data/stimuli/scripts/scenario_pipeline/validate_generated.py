"""
Phase 2: Structural Validation for Generated Scenarios
See DESIGN_PHASE2.md Section 8 for checker specification.

Each check is a single function — stack traces point directly to the problem.

Usage:
    # Validate Phase 2 generated files
    python validate_generated.py --input_dir ../llm_generated_scenarios/

    # Validate Phase 1 gold-standard files (base schema only)
    python validate_generated.py --input_dir ../human_check_scenarios/ --schema_level base

    # Custom config path
    python validate_generated.py --config ../config/generation_config.yaml
"""

import argparse
import json
import os
import re
import sys

import yaml
from difflib import SequenceMatcher


# =============================================================
# Configuration
# =============================================================

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "generation_config.yaml"
)
DEFAULT_INPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "llm_generated_scenarios"
)
DEFAULT_GOLD_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "human_check_scenarios"
)

# Files to skip during validation
SKIP_FILES = {
    "generation_metadata.json", "generation_failure_log.json",
    "goemotions_mapping.json", "severity_ratings.json",
    "acceptance_log.json",
}

# Common abbreviations that contain periods but aren't sentence endings
ABBREVIATIONS = {
    "dr.", "mr.", "mrs.", "ms.", "jr.", "sr.", "etc.", "vs.",
    "prof.", "inc.", "ltd.", "dept.", "approx.", "avg.",
}


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# =============================================================
# Check 1: Valid JSON
# =============================================================

def check_json_valid(filepath):
    """
    Parse the file as JSON. Returns (data, errors).
    If parsing fails, data is None.
    """
    try:
        with open(filepath) as f:
            data = json.load(f)
        return data, []
    except json.JSONDecodeError as e:
        return None, [f"Invalid JSON: {e}"]


# =============================================================
# Check 2: Required fields
# =============================================================

def check_required_fields(data, config, schema_level):
    """
    Verify all required top-level fields are present.
    For Phase 2 ('generated'), also checks additional required fields.
    """
    errors = []
    schema = config["base_schema"]

    for field in schema["required_top_level_fields"]:
        if field not in data:
            errors.append(f"Missing required field: '{field}'")

    if schema_level == "generated":
        gen_schema = config["generated_schema"]
        for field in gen_schema.get("additional_top_level_fields", []):
            if field not in data:
                errors.append(f"Missing generated field: '{field}'")

    return errors


# =============================================================
# Check 3: Turns structure and severity order
# =============================================================

def check_turns(data, config):
    """
    Verify exactly 3 turns with correct severity tier per turn.
    Uses turn_severity_mapping from config: turn_1→baseline, etc.
    Also checks turn-level required fields.
    """
    errors = []
    schema = config["base_schema"]
    turns = data.get("turns", [])

    # Turn count
    expected_count = schema["turns_per_scenario"]
    if len(turns) != expected_count:
        errors.append(f"Expected {expected_count} turns, got {len(turns)}")

    # Severity order
    mapping = schema["turn_severity_mapping"]
    for turn in turns:
        turn_num = turn.get("turn")
        expected_tier = mapping.get(f"turn_{turn_num}")
        actual_tier = turn.get("severity_tier")
        if expected_tier and actual_tier != expected_tier:
            errors.append(
                f"Turn {turn_num}: expected severity '{expected_tier}', "
                f"got '{actual_tier}'"
            )

    # Turn-level required fields
    for turn in turns:
        for field in schema["turn_required_fields"]:
            if field not in turn:
                errors.append(
                    f"Turn {turn.get('turn', '?')}: missing field '{field}'"
                )

    return errors


# =============================================================
# Check 4: Sentence count
# =============================================================

def count_sentences(text):
    """
    Count sentences using split-based approach.
    Handles abbreviations and ellipses better than raw punctuation counting.
    See DESIGN_PHASE2.md Section 8, Check #4 rationale.
    """
    text_lower = text.lower().strip()

    # Temporarily remove known abbreviations (word-boundary match only)
    for abbr in ABBREVIATIONS:
        pattern = r'\b' + re.escape(abbr)
        text_lower = re.sub(pattern, abbr.replace(".", ""), text_lower)

    # Remove ellipses (... or ..)
    text_lower = re.sub(r'\.{2,}', '', text_lower)

    # Split on sentence-ending punctuation followed by space or end of string
    parts = re.split(r'[.!?]+(?:\s|$)', text_lower)

    # Filter out empty parts
    sentences = [p.strip() for p in parts if p.strip()]
    return len(sentences)


def check_sentences(data, config):
    """
    Verify exactly 2 sentences per user_message.
    """
    errors = []
    expected = config["base_schema"]["sentences_per_message"]

    for turn in data.get("turns", []):
        msg = turn.get("user_message", "")
        sent_count = count_sentences(msg)
        if sent_count != expected:
            errors.append(
                f"Turn {turn.get('turn')}: expected {expected} sentences, "
                f"got {sent_count} — \"{msg[:60]}...\""
            )

    return errors


# =============================================================
# Check 5: Word count (accuracy + bounds)
# =============================================================

def check_word_count(data, filepath, config):
    """
    Check 5a: word_count field matches actual len(msg.split()).
              Auto-corrects mismatches and writes back to file.
              Mismatches are warnings, not errors.
    Check 5b: word count within bounds [min, max]. Out of bounds = error.
    """
    errors = []
    warnings = []
    bounds = config["word_count_bounds"]
    corrected = False

    for turn in data.get("turns", []):
        msg = turn.get("user_message", "")
        actual_wc = len(msg.split())

        # 5a: Accuracy (auto-correct — warning only)
        if turn.get("word_count") != actual_wc:
            warnings.append(
                f"Turn {turn.get('turn')}: word_count says "
                f"{turn.get('word_count')}, actual is {actual_wc} "
                f"(auto-corrected)"
            )
            turn["word_count"] = actual_wc
            corrected = True

        # 5b: Bounds (hard error)
        if actual_wc < bounds["min"] or actual_wc > bounds["max"]:
            errors.append(
                f"Turn {turn.get('turn')}: word count {actual_wc} outside "
                f"bounds [{bounds['min']}-{bounds['max']}]"
            )

    # Write correction back to file if needed
    if corrected:
        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError:
            pass

    return errors, warnings


# =============================================================
# Check 6: Deduplication
# =============================================================

def collect_all_prompts(dirs):
    """
    Collect all user_messages from scenario files for dedup checking.
    Scans all .json files in given directories (recursively).
    """
    prompts = []
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            continue
        for root, _, files in os.walk(dir_path):
            for f in files:
                if not f.endswith(".json") or f in SKIP_FILES:
                    continue
                try:
                    with open(os.path.join(root, f)) as fh:
                        file_data = json.load(fh)
                    if "turns" in file_data:
                        for turn in file_data["turns"]:
                            prompts.append(turn.get("user_message", ""))
                except (json.JSONDecodeError, KeyError):
                    pass
    return prompts


def check_duplicates(data, config, existing_prompts):
    """
    Flag any user_message with >threshold lexical overlap
    against all gold-standard and previously generated prompts.
    """
    errors = []
    threshold = config["dedup_threshold"]

    for turn in data.get("turns", []):
        msg = turn.get("user_message", "")
        for existing in existing_prompts:
            if existing == msg:
                continue  # Skip self-comparison
            sim = SequenceMatcher(None, msg, existing).ratio()
            if sim > threshold:
                errors.append(
                    f"Turn {turn.get('turn')}: >{threshold*100:.0f}% "
                    f"similar to existing prompt (similarity: {sim:.2f})"
                )
                break  # One match is enough to flag

    return errors


# =============================================================
# Check 7: Theme distribution
# =============================================================

def check_distribution(theme_counts):
    """
    Report theme distribution imbalances.
    Returns list of warning strings (not hard failures).
    """
    warnings = []
    if not theme_counts:
        return warnings

    counts = list(theme_counts.values())
    if len(set(counts)) > 1:
        warnings.append("Theme distribution is uneven. Consider rerunning "
                        "for underrepresented themes.")

    return warnings


# =============================================================
# Phase 2-specific checks (fixed values, metadata)
# =============================================================

def check_generated_fields(data, config):
    """
    Phase 2-only: verify fixed values and generation_metadata fields.
    """
    errors = []
    gen_schema = config["generated_schema"]

    # Top-level fixed values
    for field, expected in gen_schema.get("fixed_values", {}).items():
        actual = data.get(field)
        if actual != expected:
            errors.append(
                f"Field '{field}': expected '{expected}', got '{actual}'"
            )

    # Turn-level fixed values
    for turn in data.get("turns", []):
        for field, expected in gen_schema.get("turn_fixed_values", {}).items():
            actual = turn.get(field)
            if actual != expected:
                errors.append(
                    f"Turn {turn.get('turn')}: '{field}' expected "
                    f"'{expected}', got '{actual}'"
                )

    # generation_metadata required fields
    meta = data.get("generation_metadata", {})
    required_meta = gen_schema.get(
        "generation_metadata_fields", {}
    ).get("required", [])
    for field in required_meta:
        if field not in meta:
            errors.append(f"generation_metadata missing field: '{field}'")

    return errors


# =============================================================
# Orchestration: run all checks on one file
# =============================================================

def validate_file(filepath, config, schema_level, existing_prompts):
    """
    Run all checks on a single file.
    Returns (passed: bool, errors: list, warnings: list).
    """
    warnings = []

    # Check 1: Valid JSON
    data, errors = check_json_valid(filepath)
    if data is None:
        return False, errors, warnings

    # Check 2: Required fields
    errors += check_required_fields(data, config, schema_level)
    if errors:
        return False, errors, warnings  # Can't proceed without basic fields

    # Check 3: Turns structure and severity order
    errors += check_turns(data, config)

    # Check 4: Sentence count
    errors += check_sentences(data, config)

    # Check 5: Word count (accuracy + bounds)
    wc_errors, wc_warnings = check_word_count(data, filepath, config)
    errors += wc_errors
    warnings += wc_warnings

    # Check 6: Deduplication
    errors += check_duplicates(data, config, existing_prompts)

    # Phase 2-specific checks
    if schema_level == "generated":
        errors += check_generated_fields(data, config)

    passed = len(errors) == 0
    return passed, errors, warnings


# =============================================================
# Main validation loop
# =============================================================

class TeeWriter:
    """Write to both terminal and a log file simultaneously."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


def run_validation(input_dir, config, gold_dir, schema_level):
    """Validate all JSON files in input directory."""
    # Auto-log to input directory
    tee = TeeWriter(os.path.join(input_dir, "validation_log.txt"))
    sys.stdout = tee

    # Collect existing prompts for dedup (gold standard + already generated)
    existing_prompts = collect_all_prompts([gold_dir, input_dir])

    total = 0
    passed = 0
    failed = 0
    theme_counts = {}
    all_errors = {}

    print("=" * 60)
    print(f"Validating: {input_dir}")
    print(f"Schema level: {schema_level}")
    print("=" * 60)
    print()

    for root, dirs, files in os.walk(input_dir):
        # Skip annotation directories
        dirs[:] = [d for d in dirs if d != "annotations"]

        for f in sorted(files):
            if not f.endswith(".json") or f in SKIP_FILES:
                continue

            filepath = os.path.join(root, f)
            total += 1

            ok, errors, warnings = validate_file(
                filepath, config, schema_level, existing_prompts
            )

            # Track theme distribution
            try:
                with open(filepath) as fh:
                    theme_data = json.load(fh)
                theme = theme_data.get("theme", os.path.basename(root))
            except (json.JSONDecodeError, KeyError):
                theme = os.path.basename(root)
            theme_counts[theme] = theme_counts.get(theme, 0) + 1

            if ok:
                passed += 1
                if warnings:
                    print(f"  PASS: {f} (with {len(warnings)} warning(s))")
                    for w in warnings:
                        print(f"        ⚠ {w}")
                else:
                    print(f"  PASS: {f}")
            else:
                failed += 1
                all_errors[f] = errors
                print(f"  FAIL: {f}")
                for e in errors:
                    print(f"        → {e}")
                for w in warnings:
                    print(f"        ⚠ {w}")

    # Check 7: Theme distribution
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total:     {total}")
    print(f"Passed:    {passed}")
    print(f"Failed:    {failed}")
    if total > 0:
        rate = passed / total * 100
        print(f"Pass rate: {rate:.1f}%")
        if rate < 70:
            print("⚠️  Pass rate below 70% — consider revising "
                  "generation prompt")
    print()

    print("Theme distribution:")
    for theme, count in sorted(theme_counts.items()):
        print(f"  {theme}: {count}")

    dist_warnings = check_distribution(theme_counts)
    for w in dist_warnings:
        print(f"\n⚠️  {w}")

    # Close auto-log
    tee.close()

    # Return summary for programmatic use
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / total if total > 0 else 0,
        "theme_counts": dict(theme_counts),
        "errors": all_errors,
    }


# =============================================================
# CLI
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate generated scenario files against schema"
    )
    parser.add_argument(
        "--input_dir", type=str, default=DEFAULT_INPUT_DIR,
        help="Directory containing files to validate"
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG_PATH,
        help="Path to generation_config.yaml"
    )
    parser.add_argument(
        "--gold_dir", type=str, default=DEFAULT_GOLD_DIR,
        help="Path to gold-standard files (for dedup checking)"
    )
    parser.add_argument(
        "--schema_level", type=str, default="generated",
        choices=["base", "generated"],
        help="'base' for Phase 1 files, 'generated' for Phase 2 files"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    result = run_validation(
        input_dir=args.input_dir,
        config=config,
        gold_dir=args.gold_dir,
        schema_level=args.schema_level,
    )

    # Exit with error code if failures detected
    sys.exit(0 if result["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
