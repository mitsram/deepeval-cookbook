# deepeval-cookbook

Playground for evaluating LLM prompts with DeepEval using Anthropic Claude judges.

## Python Environment Setup

Create an isolated environment before installing dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
```

Deactivate later with `deactivate`. Delete `.venv/` to reset the environment.

## Prerequisites

- Python 3.10+
- Anthropic API key with access to Claude 3.5
- Confident AI project API key (free tier available)
- Optional: direnv or other tooling to auto-load environment variables

## Initial Setup (after cloning)

1. Activate the virtual environment (see above).
2. Install dependencies:

	```bash
	pip install -r requirements.txt
	```
3. Configure Anthropic and Confident credentials:

	```bash
	export ANTHROPIC_API_KEY="your-api-key"
	export ANTHROPIC_MODEL="claude-3-5-haiku-20241022"  # optional override
	export CONFIDENT_API_KEY="your-confident-api-key"
	```

	You can also store these in a `.env` file so `python-dotenv` loads them automatically. Copy [.env.example](.env.example) to `.env` and fill in the values.

## Project Structure

- Evaluation overview: [docs/LLM_EVALUATION.md](docs/LLM_EVALUATION.md)
- Prompt assets: [prompts/](prompts)
- DeepEval harnesses: [tests/test_requirement_analysis.py](tests/test_requirement_analysis.py), [tests/test_test_cases_table_output.py](tests/test_test_cases_table_output.py)

## Running Evaluations

With the environment active and credentials exported, run the test harnesses via Python:

```bash
deepeval test run tests/test_requirement_analysis.py
deepeval test run tests/test_test_cases_table_output.py
```

Each script loads the corresponding prompt, wraps the Anthropic client in a DeepEval-compatible model, and reports per-metric scores for the configured `LLMTestCase`.

## Confident AI Integration

- Runs now require a `CONFIDENT_API_KEY`; the tests will raise if the key is missing or if authentication fails.

## Useful Tips

- Update `prompts/` files to iterate on evaluation instructions without changing code.
- Adjust metric criteria or thresholds inside the test scripts to reflect new success criteria.
- Cache Anthropic responses during local development to reduce latency and cost when tuning prompts.