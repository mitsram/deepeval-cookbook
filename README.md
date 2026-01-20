# deepeval-cookbook

Minimal DeepEval example that scores a test-case generation prompt using an Anthropic model.

## Quick start

1. Create and activate a virtual environment (see docs/PYTHON_SETUP.md).
2. Install deps: `pip install -r requirements.txt`.
3. Export your key: `export ANTHROPIC_API_KEY=...`.
4. Optional: `export ANTHROPIC_MODEL=claude-3-5-sonnet-20241022`.
5. Run: `python evaluate_test_prompt.py`.

The script reads prompts/test_generation_prompt.md, asks Anthropic for test cases, and prints an Answer Relevancy score computed by DeepEval.