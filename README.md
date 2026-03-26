# LLM Mental Health Scenario Testing Framework

Sends mental health scenario prompts to multiple LLMs via OpenRouter and saves responses as JSON files for downstream evaluation.

## Setup

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Set your API key:**

Copy the example env file and fill in your key:
```bash
cp .env.example .env
```
Then edit `.env`:
```
OPENROUTER_API_KEY=your_openrouter_api_key
```

## Test Connection

Before running the full framework, verify your API key and model connectivity:
```bash
python test_connection.py
```
This sends a short message to each configured model and prints the status, response time, and token usage.

## How to Run

**Run all scenarios with all models:**
```bash
python main.py
```

**Preview what would run without calling the API:**
```bash
python main.py --dry-run
```

**Run specific models only:**
```bash
python main.py --models "GPT-4o,Claude Sonnet"
```

**Override repeat count:**
```bash
python main.py --repeats 1
```

**Custom paths:**
```bash
python main.py --scenarios ./my_scenarios --output ./my_output
```

**Combined example:**
```bash
python main.py --scenarios ./scenarios --models "GPT-4o" --repeats 1 --dry-run
```

## Adding New Scenarios

Place a `.json` file in the `scenarios/` directory following this format:

```json
{
  "scenario_id": "S1",
  "description": "Description of the scenario",
  "system_prompt": "Optional. Overrides the default system prompt for this scenario.",
  "turns": [
    {
      "turn": 1,
      "escalation_level": 1,
      "user_message": "User message text",
      "expected_risk": "low"
    }
  ]
}
```

The `system_prompt` field is optional. If omitted, the `default_system_prompt` from `config/settings.yaml` is used.

## Input Format

Each scenario file in `scenarios/` contains:

| Field | Type | Description |
|-------|------|-------------|
| `scenario_id` | string | Unique identifier (used in output path) |
| `description` | string | Human-readable description |
| `system_prompt` | string | Optional. Per-scenario system prompt |
| `turns` | array | List of turns to test |
| `turns[].turn` | int | Turn number |
| `turns[].escalation_level` | int | Escalation severity level |
| `turns[].user_message` | string | The message sent to the model |
| `turns[].expected_risk` | string | Expected risk level annotation |

Each turn is sent as a **standalone** single-turn request (no conversation history between turns).

## Output Format

One JSON file per `scenario × model × run`, saved to:
```
data/transcripts/{scenario_id}/{model_name}/run_{run_id}.json
```

Example output file:
```json
{
  "scenario_id": "S1",
  "model": "openai/gpt-4o",
  "model_name": "GPT-4o",
  "run_id": 1,
  "phase": "single_turn",
  "timestamp": "2025-03-28T14:30:00Z",
  "parameters": {
    "temperature": 0,
    "max_tokens": 1024,
    "system_prompt": "You are a helpful assistant."
  },
  "turns": [
    {
      "turn": 1,
      "escalation_level": 1,
      "user_message": "Some user message",
      "expected_risk": "low",
      "model_response": "The actual model response...",
      "status": "success",
      "response_time_ms": 1200,
      "usage": {
        "input_tokens": 25,
        "output_tokens": 150
      }
    }
  ],
  "total_usage": {
    "input_tokens": 25,
    "output_tokens": 150
  }
}
```

## Status Codes

| Status | Meaning |
|--------|---------|
| `success` | Normal response received |
| `refused` | Model returned empty or content-filtered response |
| `error` | API error occurred (message in `model_response`) |
| `timeout` | Request timed out after all retries |

## Configuration

- **`config/models.yaml`** — list of models to test (id + display name)
- **`config/settings.yaml`** — OpenRouter base URL, temperature, max_tokens, repeats, default system prompt

Interrupted runs resume automatically: existing output files are skipped.

## Token Usage Log

After each run, token consumption is appended to:
```
data/transcripts/usage_log.jsonl
```
Each line is one run's summary:
```json
{"timestamp": "2025-03-28T14:30:00Z", "completed": 9, "skipped": 0, "errors": 0, "input_tokens": 1234, "output_tokens": 5678, "total_tokens": 6912}
```
