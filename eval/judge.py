import json
import re

import yaml

from pipeline.api_client import call_model


def load_prompt_yaml(path: str) -> dict:
    """Load the full prompt YAML (prompts + scoring_schema)."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_judge_response(response: str) -> dict:
    """Strip markdown fences if present, then parse JSON."""
    clean = re.sub(r"```(?:json)?\s*", "", response).strip()
    clean = re.sub(r"```\s*$", "", clean).strip()
    return json.loads(clean)


def judge_turn(
    system_prompt: str,
    user_message: str,
    model_response: str,
    judge_model: str,
    judge_settings: dict,
    base_url: str,
    prompt_templates: dict,
) -> dict:
    """Evaluate one conversation turn across all dimensions.

    Supports two prompt formats:
      - template format: single template string with {{user_prompt}} / {{model_response}}
      - system/user format: separate system and user keys with {user_message} / {model_response}

    Returns a dict with keys: scores, judge_status, and optionally judge_errors.
    """
    dimensions = [
        k for k, v in prompt_templates.items()
        if isinstance(v, dict) and ("template" in v or ("system" in v and "user" in v))
    ]

    scores = {}
    errors = {}
    filtered = []

    for dim in dimensions:
        tmpl = prompt_templates[dim]

        if "template" in tmpl:
            # Original single-template format: fill {{user_prompt}} / {{model_response}}
            filled = (tmpl["template"]
                      .replace("{{user_prompt}}", user_message)
                      .replace("{{model_response}}", model_response))
            messages = [{"role": "user", "content": filled}]
        else:
            # system + user format
            user_content = tmpl["user"].format(
                system_prompt=system_prompt,
                user_message=user_message,
                model_response=model_response,
            )
            messages = [
                {"role": "system", "content": tmpl["system"]},
                {"role": "user", "content": user_content},
            ]

        result = call_model(
            model_id=judge_model,
            messages=messages,
            temperature=judge_settings.get("temperature", 0),
            max_tokens=judge_settings.get("max_tokens", 1024),
            base_url=base_url,
        )

        if result["status"] == "refused":
            filtered.append(dim)
            scores[dim] = None
            continue

        if result["status"] != "success":
            errors[dim] = result["response"]
            scores[dim] = None
            continue

        try:
            scores[dim] = _parse_judge_response(result["response"])
        except (json.JSONDecodeError, ValueError):
            scores[dim] = None

    successful = [dim for dim, s in scores.items() if s is not None]
    if len(successful) == len(dimensions):
        overall = "success"
    elif successful:
        overall = "partial"
    else:
        overall = "failed"

    out = {"scores": scores, "judge_status": overall}
    if filtered:
        out["filtered_dimensions"] = filtered
    if errors:
        out["judge_errors"] = errors
    return out
