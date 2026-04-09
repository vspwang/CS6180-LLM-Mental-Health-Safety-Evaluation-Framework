import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from pipeline.api_client import call_model
from pipeline.utils import load_json, save_json

logger = logging.getLogger(__name__)
_print_lock = threading.Lock()


def _safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def run_single_turn(scenario: dict, model_config: dict, settings: dict) -> dict:
    """Run all turns of a scenario independently (single-turn mode) against one model."""
    system_prompt = scenario.get(
        "system_prompt", settings["default_system_prompt"]
    )
    base_url = settings["base_url"]
    temperature = settings["temperature"]
    max_tokens = settings["max_tokens"]

    turns_output = []
    total_input_tokens = 0
    total_output_tokens = 0

    for turn in scenario["turns"]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": turn["user_message"]},
        ]

        result = call_model(
            model_id=model_config["id"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
        )

        total_input_tokens += result["usage"]["input_tokens"]
        total_output_tokens += result["usage"]["output_tokens"]

        turns_output.append(
            {
                "turn": turn["turn"],
                "severity_tier": turn.get("severity_tier"),
                "data_source": turn.get("data_source"),
                "user_message": turn["user_message"],
                "word_count": turn.get("word_count"),
                "model_response": result["response"],
                "status": result["status"],
                "response_time_ms": result["response_time_ms"],
                "usage": result["usage"],
            }
        )

    return {
        "stimulus_id": scenario["stimulus_id"],
        "theme": scenario.get("theme"),
        "variant": scenario.get("variant"),
        "goemotions_categories": scenario.get("goemotions_categories"),
        "model": model_config["id"],
        "model_name": model_config["name"],
        "run_id": None,  # filled in by run_batch
        "phase": "single_turn",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parameters": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
        },
        "turns": turns_output,
        "total_usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        },
    }


def _run_one(
    stimulus_path: Path,
    model_config: dict,
    run_id: int,
    settings: dict,
    output_dir: str,
) -> tuple[str, dict | None]:
    """Run one (stimulus, model, run) combination. Returns (status, data)."""
    scenario = load_json(str(stimulus_path))
    stimulus_id = scenario["stimulus_id"]
    model_name = model_config["name"]
    model_slug = model_name.replace(" ", "")

    out_path = (
        Path(output_dir)
        / stimulus_id
        / f"transcript_{stimulus_id}_{model_slug}.json"
    )

    if out_path.exists():
        existing = load_json(str(out_path))
        has_errors = any(
            t.get("status") in ("error", "timeout")
            for t in existing.get("turns", [])
        )
        if not has_errors:
            return "skipped", None

    repeats = settings["repeats"]
    _safe_print(f"Running {stimulus_id} on {model_name}, run {run_id}/{repeats}...")

    try:
        transcript = run_single_turn(scenario, model_config, settings)
        transcript["run_id"] = run_id
        save_json(str(out_path), transcript)
        run_errors = sum(
            1 for t in transcript["turns"] if t["status"] in ("error", "timeout")
        )
        return "completed", {"transcript": transcript, "errors": run_errors}
    except Exception as e:
        logger.error("Failed %s / %s / run %d: %s", stimulus_id, model_name, run_id, e)
        return "error", None


def run_batch(
    stimuli_files: list,
    models: list,
    settings: dict,
    output_dir: str,
    max_workers: int = 8,
) -> None:
    if not stimuli_files:
        logger.error("No stimulus files provided")
        return

    repeats = settings["repeats"]

    tasks = [
        (stimulus_path, model_config, run_id)
        for stimulus_path in stimuli_files
        for model_config in models
        for run_id in range(1, repeats + 1)
    ]

    total_completed = 0
    total_skipped = 0
    total_errors = 0
    total_input_tokens = 0
    total_output_tokens = 0
    counter_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_one, sp, mc, rid, settings, output_dir): (sp, mc, rid)
            for sp, mc, rid in tasks
        }
        for future in as_completed(futures):
            status, data = future.result()
            with counter_lock:
                if status == "skipped":
                    total_skipped += 1
                elif status == "completed":
                    total_completed += 1
                    total_errors += data["errors"]
                    total_input_tokens += data["transcript"]["total_usage"]["input_tokens"]
                    total_output_tokens += data["transcript"]["total_usage"]["output_tokens"]
                else:
                    total_errors += 1

    print("\n--- Summary ---")
    print(f"Completed : {total_completed}")
    print(f"Skipped   : {total_skipped}")
    print(f"Errors    : {total_errors}")
    print(
        f"Tokens    : {total_input_tokens} input / {total_output_tokens} output "
        f"({total_input_tokens + total_output_tokens} total)"
    )

    _append_usage_log(
        output_dir=output_dir,
        completed=total_completed,
        skipped=total_skipped,
        errors=total_errors,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
    )


def _append_usage_log(
    output_dir: str,
    completed: int,
    skipped: int,
    errors: int,
    input_tokens: int,
    output_tokens: int,
) -> None:
    log_path = Path(output_dir) / "usage_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "completed": completed,
        "skipped": skipped,
        "errors": errors,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Usage log : {log_path}")
