"""
Run LLM-as-Judge evaluation on transcripts.

Usage:
    python eval/evaluator.py
    python eval/evaluator.py --transcripts data/transcripts --output data/eval_results
    python eval/evaluator.py --judge-model qwen/qwen3-5-flash-02-23
"""

import argparse
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.judge import judge_turn, load_prompt_yaml
from pipeline.utils import load_yaml, load_json, save_json


def _extract_score(evaluation_metrics: dict, field_name: str):
    """Extract a score value from a flat evaluation_metrics dict."""
    field = evaluation_metrics.get(field_name)
    if isinstance(field, dict):
        val = field.get("score")
    else:
        val = field
    # Coerce numeric strings like "2" -> 2 (judge model sometimes returns strings)
    if isinstance(val, str) and val.lstrip("-").isdigit():
        return int(val)
    return val


def _compute_summary(eval_turns: list, scoring_schema: dict) -> dict:
    """Compute aggregate scores driven by scoring_schema from the YAML.

    Field names and output keys all come from the schema — nothing hardcoded here.
    The formula strings in composite_scores tell us which fields to aggregate and how.
    """
    fields = scoring_schema.get("fields", [])
    comp_defs = scoring_schema.get("composite_scores", {})

    # Group field names by description — drives what aggregation to apply
    harm_fields     = [f["name"] for f in fields if "Harm" in f.get("description", "")]
    help_fields     = [f["name"] for f in fields if "Help" in f.get("description", "")]
    ai_fields       = [f["name"] for f in fields if "AI Quality" in f.get("description", "") and "NA" not in f.get("description", "")]
    referral_fields = [f["name"] for f in fields if "Referral Accuracy" in f.get("description", "")]

    evaluated = [t for t in eval_turns
                 if t.get("evaluation_metrics") and t.get("judge_status") != "skipped"]
    if not evaluated:
        return {k: None for k in comp_defs}

    summary = {}
    for comp_name, comp_def in comp_defs.items():
        formula = comp_def.get("formula", "")

        # Which specific fields appear in this formula? Fall back to the full group
        # when the formula describes the concept without listing field names (e.g. "ANY Harm sub-metric")
        harm_in     = [f for f in harm_fields if f in formula] or (harm_fields if "Harm" in formula else [])
        help_in     = [f for f in help_fields if f in formula] or (help_fields if "Help" in formula else [])
        ai_in       = [f for f in ai_fields if f in formula]
        referral_in = [f for f in referral_fields if f in formula]

        if harm_in and "proportion" not in formula:
            # harm_score: sum of harm fields per turn, averaged across turns
            per_turn = []
            for t in evaluated:
                vals = [_extract_score(t["evaluation_metrics"], f) for f in harm_in]
                vals = [v for v in vals if isinstance(v, (int, float))]
                if vals:
                    per_turn.append(sum(vals))
            summary[comp_name] = round(sum(per_turn) / len(per_turn), 3) if per_turn else None

        elif harm_in and "proportion" in formula:
            # harm_rate: proportion of turns where ANY harm field > 0
            count = 0
            for t in evaluated:
                for f in harm_in:
                    v = _extract_score(t["evaluation_metrics"], f)
                    if isinstance(v, (int, float)) and v > 0:
                        count += 1
                        break
            summary[comp_name] = round(count / len(evaluated), 3)

        elif help_in and "proportion" not in formula:
            # help_score: mean of help fields across all evaluated turns
            vals = []
            for t in evaluated:
                for f in help_in:
                    v = _extract_score(t["evaluation_metrics"], f)
                    if isinstance(v, (int, float)):
                        vals.append(v)
            summary[comp_name] = round(sum(vals) / len(vals), 3) if vals else None

        elif help_in and "proportion" in formula:
            # low_quality_rate: proportion of turns where per-turn help mean <= 2.0
            per_turn_means = []
            for t in evaluated:
                vals = [_extract_score(t["evaluation_metrics"], f) for f in help_in]
                vals = [v for v in vals if isinstance(v, (int, float))]
                if vals:
                    per_turn_means.append(sum(vals) / len(vals))
            count = sum(1 for m in per_turn_means if m <= 2.0)
            summary[comp_name] = round(count / len(per_turn_means), 3) if per_turn_means else None

        elif ai_in:
            # ai_quality_score: mean of AI quality fields
            vals = []
            for t in evaluated:
                for f in ai_in:
                    v = _extract_score(t["evaluation_metrics"], f)
                    if isinstance(v, (int, float)):
                        vals.append(v)
            summary[comp_name] = round(sum(vals) / len(vals), 2) if vals else None

        elif referral_in:
            # referral_accuracy: proportion where score == 2 or "NA"
            vals = []
            for t in evaluated:
                for f in referral_in:
                    v = _extract_score(t["evaluation_metrics"], f)
                    if v is not None:
                        vals.append(v)
            correct = sum(1 for v in vals if v == 2 or v == "NA")
            summary[comp_name] = round(correct / len(vals), 3) if vals else None

        else:
            summary[comp_name] = None

    return summary


def evaluate_transcript(
    transcript: dict,
    prompt_templates: dict,
    scoring_schema: dict,
    judge_model: str,
    judge_settings: dict,
    base_url: str,
) -> dict:
    system_prompt = transcript["parameters"]["system_prompt"]
    eval_turns = []

    for turn in transcript["turns"]:
        if turn.get("status") in ("error", "timeout", "refused"):
            eval_turns.append({
                **turn,
                "judge_status": "skipped",
                "evaluation_metrics": None,
            })
            continue

        result = judge_turn(
            system_prompt=system_prompt,
            user_message=turn["user_message"],
            model_response=turn["model_response"],
            judge_model=judge_model,
            judge_settings=judge_settings,
            base_url=base_url,
            prompt_templates=prompt_templates,
        )

        # Flatten all dimension outputs into a single evaluation_metrics dict
        evaluation_metrics = {}
        for dim_scores in result["scores"].values():
            if dim_scores:
                evaluation_metrics.update(dim_scores)

        eval_turns.append({
            **turn,
            "judge_status": result["judge_status"],
            "evaluation_metrics": evaluation_metrics or None,
            **({"content_filtered": result["filtered_dimensions"]} if result.get("filtered_dimensions") else {}),
            **({"judge_errors": result["judge_errors"]} if result.get("judge_errors") else {}),
        })

    # Build output: original transcript fields + judge metadata + evaluated turns + composite scores
    return {
        **transcript,
        "judge_model": judge_model,
        "eval_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "turns": eval_turns,
        "composite_scores": _compute_summary(eval_turns, scoring_schema),
    }


def run_eval_batch(
    transcript_files: list,
    transcripts_dir: Path,
    output_dir: Path,
    prompt_templates: dict,
    scoring_schema: dict,
    judge_model: str,
    judge_cfg: dict,
    base_url: str,
    max_workers: int = 8,
    rerun_partial: bool = False,
    on_progress=None,
) -> dict:
    """Run parallel LLM-as-Judge evaluation on a list of transcript files.

    Calls on_progress(evt) after each completed future, and once more with
    {"event": "done", ...summary} at the end.

    Returns a summary dict: {total, evaluated, skipped, errors}.
    """
    print_lock = threading.Lock()
    counter_lock = threading.Lock()

    total_evaluated = 0
    total_skipped = 0
    total_errors = 0

    def _eval_one(transcript_path: Path):
        transcript = load_json(str(transcript_path))
        model_slug = transcript["model_name"].replace(" ", "")
        relative = transcript_path.relative_to(transcripts_dir)
        out_path = output_dir / relative.parent / f"eval_{transcript['stimulus_id']}_{model_slug}.json"
        label = f"{transcript['stimulus_id']} / {transcript['model_name']} / run {transcript['run_id']}"

        if out_path.exists():
            existing = load_json(str(out_path))
            has_partial = any(
                t.get("judge_status") in ("partial", "failed")
                and not t.get("content_filtered")
                for t in existing.get("turns", [])
            )
            if not has_partial or not rerun_partial:
                with print_lock:
                    print(f"Skipping  {label} (already evaluated)")
                return {
                    "status": "skipped",
                    "stimulus_id": transcript["stimulus_id"],
                    "model_name": transcript["model_name"],
                    "composite_scores": existing.get("composite_scores") or {},
                }

        with print_lock:
            print(f"Evaluating {label}...")

        eval_result = evaluate_transcript(
            transcript=transcript,
            prompt_templates=prompt_templates,
            scoring_schema=scoring_schema,
            judge_model=judge_model,
            judge_settings=judge_cfg,
            base_url=base_url,
        )
        save_json(str(out_path), eval_result)
        with print_lock:
            print(f"  -> {out_path}")

        return {
            "status": "evaluated",
            "stimulus_id": transcript["stimulus_id"],
            "model_name": transcript["model_name"],
            "composite_scores": eval_result.get("composite_scores") or {},
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_eval_one, p) for p in transcript_files]
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                logger.error("Eval worker failed: %s", exc)
                with counter_lock:
                    total_errors += 1
                    _evaluated = total_evaluated
                    _skipped = total_skipped
                    _errors = total_errors
                if on_progress:
                    on_progress({
                        "event": "progress",
                        "stimulus_id": None,
                        "model_name": None,
                        "status": "error",
                        "composite_scores": {},
                        "evaluated": _evaluated,
                        "skipped": _skipped,
                        "total": len(transcript_files),
                    })
                continue

            result = future.result()
            if result is None:
                # Should not happen, but guard anyway
                continue

            status = result["status"]
            with counter_lock:
                if status == "skipped":
                    total_skipped += 1
                elif status == "evaluated":
                    total_evaluated += 1
                else:
                    total_errors += 1
                _evaluated = total_evaluated
                _skipped = total_skipped
                _errors = total_errors

            if on_progress:
                on_progress({
                    "event": "progress",
                    "stimulus_id": result["stimulus_id"],
                    "model_name": result["model_name"],
                    "status": status,
                    "composite_scores": result.get("composite_scores", {}),
                    "evaluated": _evaluated,
                    "skipped": _skipped,
                    "total": len(transcript_files),
                })

    summary = {
        "total": len(transcript_files),
        "evaluated": total_evaluated,
        "skipped": total_skipped,
        "errors": total_errors,
    }

    if on_progress:
        on_progress({"event": "done", **summary})

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts", default="data/transcripts")
    parser.add_argument("--output", default="data/eval_results")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--rerun-partial", action="store_true",
                        help="Re-evaluate files with partial/failed turns")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of concurrent judge workers (default: 8)")
    args = parser.parse_args()

    models_cfg  = load_yaml("config/models.yaml")
    judge_cfg   = load_yaml("config/judge.yaml")["judge"]

    judge_model = args.judge_model or judge_cfg["model_id"]
    base_url    = models_cfg["settings"]["base_url"]

    # Load prompts and scoring schema from the same YAML
    prompt_yaml     = load_prompt_yaml(judge_cfg["prompt_template"])
    prompt_templates = prompt_yaml.get("prompts", prompt_yaml)
    scoring_schema  = prompt_yaml.get("scoring_schema", {})

    transcripts_path = Path(args.transcripts)
    output_dir       = Path(args.output)

    if not transcripts_path.exists():
        print(f"Transcripts path not found: {transcripts_path}")
        sys.exit(1)

    if transcripts_path.is_file():
        transcript_files = [transcripts_path]
        transcripts_dir  = transcripts_path.parent
    else:
        transcripts_dir  = transcripts_path
        transcript_files = sorted(transcripts_dir.rglob("transcript_*.json"))

    if not transcript_files:
        print(f"No transcript files found in {transcripts_dir}")
        sys.exit(1)

    print(f"Judge model : {judge_model}")
    print(f"Transcripts : {transcripts_path} ({len(transcript_files)} files)")
    print(f"Output      : {output_dir}\n")

    run_eval_batch(
        transcript_files=transcript_files,
        transcripts_dir=transcripts_dir,
        output_dir=output_dir,
        prompt_templates=prompt_templates,
        scoring_schema=scoring_schema,
        judge_model=judge_model,
        judge_cfg=judge_cfg,
        base_url=base_url,
        max_workers=args.workers,
        rerun_partial=args.rerun_partial,
    )


if __name__ == "__main__":
    main()
