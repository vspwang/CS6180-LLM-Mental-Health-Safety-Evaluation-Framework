import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.api_client import call_model
from src.utils import load_json, save_json

logger = logging.getLogger(__name__)


def run_single_turn(scenario: dict, model_config: dict, settings: dict) -> dict:
    """Run all turns of a scenario independently (single-turn mode) against one model."""
    system_prompt = scenario.get(
        "system_prompt", settings["test"]["default_system_prompt"]
    )
    base_url = settings["openrouter"]["base_url"]
    temperature = settings["test"]["temperature"]
    max_tokens = settings["test"]["max_tokens"]

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
                "escalation_level": turn["escalation_level"],
                "user_message": turn["user_message"],
                "expected_risk": turn["expected_risk"],
                "model_response": result["response"],
                "status": result["status"],
                "response_time_ms": result["response_time_ms"],
                "usage": result["usage"],
            }
        )

    return {
        "scenario_id": scenario["scenario_id"],
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


def run_batch(
    scenarios_dir: str,
    models: list,
    settings: dict,
    output_dir: str,
) -> None:
    scenario_files = sorted(Path(scenarios_dir).glob("*.json"))
    if not scenario_files:
        logger.error("No .json scenario files found in %s", scenarios_dir)
        return

    repeats = settings["test"]["repeats"]

    total_completed = 0
    total_skipped = 0
    total_errors = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for scenario_path in scenario_files:
        scenario = load_json(str(scenario_path))
        scenario_id = scenario["scenario_id"]

        for model_config in models:
            model_name = model_config["name"]

            for run_id in range(1, repeats + 1):
                out_path = (
                    Path(output_dir)
                    / scenario_id
                    / model_name
                    / f"run_{run_id}.json"
                )

                if out_path.exists():
                    total_skipped += 1
                    continue

                print(
                    f"Running {scenario_id} on {model_name}, run {run_id}/{repeats}..."
                )

                try:
                    transcript = run_single_turn(scenario, model_config, settings)
                    transcript["run_id"] = run_id

                    save_json(str(out_path), transcript)

                    run_errors = sum(
                        1 for t in transcript["turns"] if t["status"] in ("error", "timeout")
                    )
                    total_errors += run_errors
                    total_input_tokens += transcript["total_usage"]["input_tokens"]
                    total_output_tokens += transcript["total_usage"]["output_tokens"]
                    total_completed += 1

                except Exception as e:
                    logger.error(
                        "Failed %s / %s / run %d: %s", scenario_id, model_name, run_id, e
                    )
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
