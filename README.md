# LLM Mental Health Response Evaluation Framework

A two-stage framework for evaluating how LLMs respond to users expressing emotional distress. Stage 1 collects model responses to structured stimuli; Stage 2 runs an LLM-as-Judge evaluation across three dimensions (harm, help, AI quality).

## Project Structure

```
├── config/
│   ├── models.yaml          # Models to test + judge model
│   ├── settings.yaml        # API settings, temperature, repeats
│   └── eval_settings.yaml   # Judge settings + prompt template path
├── data/
│   ├── stimuli/             # Test case input files (.json)
│   ├── transcripts/         # Stage 1 output: model responses
│   └── eval_results/        # Stage 2 output: evaluation scores
├── eval/
│   ├── judge.py             # LLM-as-Judge logic
│   ├── run_eval.py          # Evaluation entry point
│   └── prompts/
│       └── eval_prompts.yaml  # Judge prompts + scoring schema
├── pipeline/
│   ├── api_client.py        # OpenRouter API wrapper
│   ├── test_runner.py       # Stage 1 runner
│   └── utils.py             # Shared utilities
├── main.py                  # Stage 1 entry point
└── test_connection.py       # API connectivity check
```

## Setup

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Set your API key:**
```bash
cp .env.example .env
# Edit .env and fill in: OPENROUTER_API_KEY=your_key
```

**Verify connectivity:**
```bash
python test_connection.py
```

---

## Stage 1: Collect Model Responses

Sends each stimulus to all configured models and saves responses as transcript files.

**Run all stimuli with all models:**
```bash
python main.py
```

**Preview without calling the API:**
```bash
python main.py --dry-run
```

**Other options:**
```bash
python main.py --models "GPT-5.4 Nano,DeepSeek V3.2"   # specific models
python main.py --repeats 1                               # override repeat count
python main.py --stimuli data/stimuli/s1.json           # single stimulus file
python main.py --stimuli data/stimuli --output data/transcripts  # custom paths
```

### Adding New Stimuli

Place a `.json` file in `data/stimuli/` with this format:

```json
{
  "stimulus_id": "S1",
  "description": "Brief description of the test case",
  "system_prompt": "Optional. Overrides the default system prompt.",
  "turns": [
    {
      "turn": 1,
      "escalation_level": 1,
      "user_message": "I've been feeling a bit tired lately.",
      "expected_risk": "low"
    }
  ]
}
```

`system_prompt` is optional — if omitted, the default from `config/settings.yaml` is used. Each turn is sent as a **standalone** single-turn request (no conversation history).

### Stage 1 Output

One transcript per `stimulus × model`, saved to:
```
data/transcripts/{stimulus_id}/{model_name}/transcript_{stimulus_id}_{model_name}.json
```

```json
{
  "stimulus_id": "S1",
  "model": "openai/gpt-5.4-nano",
  "model_name": "GPT-5.4 Nano",
  "run_id": 1,
  "parameters": { "temperature": 0, "max_tokens": 1024, "system_prompt": "..." },
  "turns": [
    {
      "turn": 1,
      "escalation_level": 1,
      "user_message": "...",
      "expected_risk": "low",
      "model_response": "...",
      "status": "success",
      "response_time_ms": 1200,
      "usage": { "input_tokens": 25, "output_tokens": 150 }
    }
  ],
  "total_usage": { "input_tokens": 25, "output_tokens": 150 }
}
```

---

## Stage 2: Evaluate Responses

Runs an LLM-as-Judge over all transcripts. Each turn is evaluated with 3 separate API calls — one per dimension.

```bash
python eval/run_eval.py
```

**Options:**
```bash
python eval/run_eval.py --transcripts data/transcripts --output data/eval_results
python eval/run_eval.py --transcripts data/transcripts/S1/GPT-5.4Nano/transcript_S1_GPT-5.4Nano.json  # single file
python eval/run_eval.py --judge-model anthropic/claude-sonnet-4-6
```

### Evaluation Dimensions

| Dimension | Metrics | Scale |
|-----------|---------|-------|
| **Harm** | Emotional invalidation, Maladaptive reinforcement | 0–2 (lower = better) |
| **Help** | Emotional validation, Actionable support, Cognitive reframing | 1–5 (higher = better) |
| **AI Quality** | Boundary appropriateness, Professional referral | 1–5 / 0–2–NA |

### Stage 2 Output

One eval file per transcript, saved to:
```
data/eval_results/{stimulus_id}/{model_name}/eval_{stimulus_id}_{model_name}.json
```

Each turn's raw judge scores are stored under `evaluation_metrics`, and aggregate stats are in `composite_scores`:

```json
{
  "stimulus_id": "S1",
  "model_name": "GPT-5.4 Nano",
  "judge_model": "...",
  "turns": [
    {
      "turn": 1,
      "user_message": "...",
      "model_response": "...",
      "judge_status": "success",
      "evaluation_metrics": {
        "emotional_invalidation": { "score": 0, "reasoning": "..." },
        "maladaptive_reinforcement": { "score": 0, "reasoning": "..." },
        "error_tags": [],
        "emotional_validation": { "score": 4, "reasoning": "..." },
        "actionable_support": { "score": 4, "reasoning": "..." },
        "cognitive_reframing": { "score": 3, "reasoning": "..." },
        "boundary_appropriateness": { "score": 4, "reasoning": "..." },
        "professional_referral": { "score": "NA", "reasoning": "..." }
      }
    }
  ],
  "composite_scores": {
    "harm_score": 0.0,
    "harm_rate": 0.0,
    "help_score": 3.89,
    "low_quality_rate": 0.0,
    "ai_quality_score": 4.5,
    "referral_accuracy": 1.0
  }
}
```

---

## Configuration

| File | Purpose |
|------|---------|
| `config/models.yaml` | Models to test and judge model ID |
| `config/settings.yaml` | API base URL, temperature, max_tokens, repeats, default system prompt |
| `config/eval_settings.yaml` | Judge temperature, max_tokens, prompt template path |
| `eval/prompts/eval_prompts.yaml` | Judge prompts (one per dimension) + scoring schema |

## Status Codes

| Status | Meaning |
|--------|---------|
| `success` | Normal response received |
| `refused` | Model returned empty or content-filtered response |
| `error` | API error |
| `timeout` | Request timed out |
