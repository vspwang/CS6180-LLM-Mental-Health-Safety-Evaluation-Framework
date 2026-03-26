import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # loads .env file if present

from src.utils import load_yaml

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(
        description="LLM mental health scenario testing framework"
    )
    parser.add_argument(
        "--scenarios",
        default="./scenarios",
        help="Path to scenarios directory (default: ./scenarios)",
    )
    parser.add_argument(
        "--output",
        default="./data/transcripts",
        help="Path to output directory (default: ./data/transcripts)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model names to run (default: all models in config)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Override repeat count (default: from settings.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without calling the API",
    )
    args = parser.parse_args()

    # Load config
    config_dir = Path(__file__).parent / "config"
    settings = load_yaml(str(config_dir / "settings.yaml"))
    models_config = load_yaml(str(config_dir / "models.yaml"))["models"]

    # Apply overrides
    if args.repeats is not None:
        settings["test"]["repeats"] = args.repeats

    # Filter models if --models specified
    if args.models:
        requested = {m.strip() for m in args.models.split(",")}
        models = [m for m in models_config if m["name"] in requested]
        if not models:
            print(
                f"Error: none of the requested model names ({args.models}) found in config.",
                file=sys.stderr,
            )
            print(
                "Available models: " + ", ".join(m["name"] for m in models_config),
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        models = models_config

    scenarios_dir = Path(args.scenarios)
    output_dir = Path(args.output)
    repeats = settings["test"]["repeats"]

    # Validate scenarios directory
    if not scenarios_dir.exists() or not scenarios_dir.is_dir():
        print(f"Error: scenarios directory not found: {scenarios_dir}", file=sys.stderr)
        sys.exit(1)

    scenario_files = sorted(scenarios_dir.glob("*.json"))
    if not scenario_files:
        print(
            f"Error: no .json files found in {scenarios_dir}", file=sys.stderr
        )
        sys.exit(1)

    if args.dry_run:
        _print_dry_run(scenario_files, models, repeats, output_dir)
        return

    # Validate API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "Error: OPENROUTER_API_KEY environment variable is not set.\n"
            "Set it with: export OPENROUTER_API_KEY=your_key",
            file=sys.stderr,
        )
        sys.exit(1)

    from src.test_runner import run_batch

    run_batch(
        scenarios_dir=str(scenarios_dir),
        models=models,
        settings=settings,
        output_dir=str(output_dir),
    )


def _print_dry_run(scenario_files, models, repeats, output_dir):
    from src.utils import load_json

    total_calls = 0
    would_run = 0
    would_skip = 0

    for scenario_path in scenario_files:
        scenario = load_json(str(scenario_path))
        scenario_id = scenario["scenario_id"]

        for model_config in models:
            model_name = model_config["name"]

            for run_id in range(1, repeats + 1):
                out_path = output_dir / scenario_id / model_name / f"run_{run_id}.json"
                exists = out_path.exists()
                status = "SKIP (exists)" if exists else "WOULD RUN"
                print(
                    f"  [{status}] {scenario_id} x {model_name} x run {run_id} "
                    f"-> {out_path}"
                )
                total_calls += 1
                if exists:
                    would_skip += 1
                else:
                    would_run += 1

    print(f"\nTotal combinations : {total_calls}")
    print(f"Would run          : {would_run}")
    print(f"Would skip         : {would_skip}")
    print(f"API calls needed   : {would_run}")


if __name__ == "__main__":
    main()
