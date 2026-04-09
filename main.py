import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # loads .env file if present

from pipeline.utils import load_yaml

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(
        description="LLM mental health response evaluation framework — Stage 1"
    )
    parser.add_argument(
        "--stimuli",
        default="data/stimuli",
        help="Stimulus file or directory (default: data/stimuli)",
    )
    parser.add_argument(
        "--output",
        default="data/transcripts",
        help="Output directory for transcripts (default: data/transcripts)",
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
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent API workers (default: 8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without calling the API",
    )
    args = parser.parse_args()

    # Load config
    config_dir = Path(__file__).parent / "config"
    models_yaml = load_yaml(str(config_dir / "models.yaml"))
    settings = models_yaml["settings"]
    models_config = models_yaml["models"]

    # Apply overrides
    if args.repeats is not None:
        settings["repeats"] = args.repeats

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

    # Accept a single stimulus file or a directory of stimulus files
    stimuli_path = Path(args.stimuli)
    if stimuli_path.is_file():
        stimuli_files = [stimuli_path]
    elif stimuli_path.is_dir():
        _non_stimulus = {"goemotions_mapping.json", "generation_metadata.json"}
        stimuli_files = sorted(
            f for f in stimuli_path.rglob("*.json")
            if "annotations" not in f.parts
            and f.name not in _non_stimulus
        )
        if not stimuli_files:
            print(f"Error: no stimulus files found in {stimuli_path}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Error: not found: {stimuli_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    repeats = settings["repeats"]

    if args.dry_run:
        _print_dry_run(stimuli_files, models, repeats, output_dir)
        return

    # Validate API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "Error: OPENROUTER_API_KEY environment variable is not set.\n"
            "Set it with: export OPENROUTER_API_KEY=your_key",
            file=sys.stderr,
        )
        sys.exit(1)

    from pipeline.test_runner import run_batch

    run_batch(
        stimuli_files=stimuli_files,
        models=models,
        settings=settings,
        output_dir=str(output_dir),
        max_workers=args.workers,
    )


def _print_dry_run(stimuli_files, models, repeats, output_dir):
    from pipeline.utils import load_json

    total_calls = 0
    would_run = 0
    would_skip = 0

    for stimulus_path in stimuli_files:
        stimulus = load_json(str(stimulus_path))
        stimulus_id = stimulus["stimulus_id"]

        for model_config in models:
            model_name = model_config["name"]
            model_slug = model_name.replace(" ", "")

            for run_id in range(1, repeats + 1):
                out_path = (
                    output_dir / stimulus_id
                    / f"transcript_{stimulus_id}_{model_slug}.json"
                )
                exists = out_path.exists()
                status = "SKIP (exists)" if exists else "WOULD RUN"
                print(f"  [{status}] {stimulus_id} x {model_name} -> {out_path}")
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
