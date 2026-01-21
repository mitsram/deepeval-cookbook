# LLM-Based Evaluation with DeepEval

## Conceptual Overview

Large Language Model (LLM) evaluation verifies that model outputs satisfy product requirements, stay safe, and remain consistent over time. A robust evaluation program combines:

- **Intrinsic quality** checks: correctness, clarity, relevance, completeness, and efficiency of responses.
- **Risk controls**: bias, fairness, safety, and policy compliance.
- **User-aligned success criteria**: task-specific expectations defined by product owners.

DeepEval provides a flexible abstraction for defining tasks, curating metrics, and executing evaluations against live LLMs. In this repository it wraps Anthropic Claude models so you can score prompts offline before shipping them to production.

## Repository Evaluation Assets

- Harness for requirement analysis prompts: [tests/test_requirement_analysis.py](../tests/test_requirement_analysis.py)
- Harness for tabular test-case prompts: [tests/test_test_cases_table_output.py](../tests/test_test_cases_table_output.py)
- Prompt bank for QA coverage: [prompts/test_generation_prompt.md](../prompts/test_generation_prompt.md)
- Prompt bank for requirement reviews: [prompts/requirement_analysis.md](../prompts/requirement_analysis.md)
- Prompt bank for table outputs: [prompts/test_cases_table_output.md](../prompts/test_cases_table_output.md)
- Environment bootstrap instructions: [README.md](../README.md#python-environment-setup)
- Dependency manifest: [requirements.txt](../requirements.txt)

These files illustrate how to: load prompts, wrap a hosted Claude model in a DeepEval-compatible interface, declare `GEval` metrics, and run `evaluate` across curated `LLMTestCase` instances.

## DeepEval Building Blocks

| Component | Purpose | Where Implemented |
|-----------|---------|-------------------|
| `LLMTestCase` | Captures the evaluation input, actual model output, and reference expectation. | [tests/test_requirement_analysis.py](../tests/test_requirement_analysis.py), [tests/test_test_cases_table_output.py](../tests/test_test_cases_table_output.py) |
| `DeepEvalBaseLLM` subclass | Adapts external LLM client APIs (Anthropic) to DeepEval. | `ClaudeModel` implementations in each harness |
| `GEval` metric | Expresses judging criteria via natural-language instructions and scored thresholds. | Metric definitions in [tests/test_requirement_analysis.py](../tests/test_requirement_analysis.py), [tests/test_test_cases_table_output.py](../tests/test_test_cases_table_output.py) |
| `evaluate` runner | Executes metrics against test cases and aggregates results. | Tail sections of the same harness files |

DeepEval executes each metric by prompting the chosen judge model (Claude) with context fragments aligned to `evaluation_params`, then comparing numeric scores against provided thresholds.

## Process Lifecycle

1. **Define the evaluation goal**
   - Clarify the user scenario and risk posture (e.g., medical advice triage vs. requirement rewriting).
   - Derive measurable success criteria (accuracy, coverage, tone compliance).
2. **Curate canonical prompts and references**
   - Draft representative inputs in the `prompts` directory.
   - Craft gold-standard responses (`expected_output`) that embody desired behavior.
3. **Design metrics and thresholds**
   - Start with reusable `GEval` patterns (correctness, clarity, relevance).
   - Add task-specific criteria by adjusting the `criteria` text and `evaluation_params`.
   - Set thresholds based on acceptable risk tolerance (0.5 minimum in the examples).
4. **Integrate judge models**
  - Implement or extend `DeepEvalBaseLLM` to call the judge provider (Claude, OpenAI, etc.).
  - Externalize provider credentials through environment variables (`ANTHROPIC_API_KEY`, `CONFIDENT_API_KEY`) or a `.env` file loaded by `python-dotenv`.
  - When `CONFIDENT_API_KEY` is present the harnesses call `deepeval.login`, pushing evaluation telemetry to Confident AI dashboards automatically.
  - The harnesses now stop execution if `CONFIDENT_API_KEY` is missing or authentication fails to guarantee Confident uploads happen.
5. **Run evaluations**
   - Execute targeted scripts locally:
     ```bash
     # Requirement analysis evaluation
     python tests/test_requirement_analysis.py

     # Tabular test-case generation evaluation
     python tests/test_test_cases_table_output.py
     ```
   - Inspect per-metric scores and failure details in the CLI output.
6. **Analyze and iterate**
   - Investigate failing metrics to refine prompts, completions, or requirements.
   - Introduce additional metrics (bias, hallucination, toxicity) as new risks emerge.
7. **Operationalize**
   - Automate runs in CI/CD.
   - Gate releases based on score regressions versus baseline snapshots.
   - Store score artifacts for longitudinal tracking.

## Key Activities

- **Data specification**
  - Expand the prompt catalog with edge cases, policy violations, and regression tests.
  - Version test cases so you can reproduce historic evaluations.
- **Metric authoring**
  - Standardize language templates for clarity, relevance, accuracy, and safety.
  - Calibrate thresholds by sampling multiple model outputs and reviewing judge stability.
- **Judge management**
  - Monitor latency and cost of the Anthropic client; switch models (Sonnet vs. Haiku) as needed.
  - Cache judge responses during local experimentation to stabilize metrics.
- **Evaluation execution**
  - Batch test cases to control API usage.
  - Collect raw judge rationales (`evaluation.get_reasoning()`) when debugging borderline scores.
- **Result interpretation**
  - Identify systemic failures (e.g., recurring clarity misses) and feed insights back into prompt engineering.
  - Track score deltas against previous runs before accepting a release.
- **Automation & governance**
  - Schedule nightly evaluations against production prompts.
  - Persist results to dashboards or experiment tracking tools for auditability.
  - Implement alerting when critical metrics (bias, accuracy) fall below contractual thresholds.

## Extending the Cookbook

- **Add multi-case suites**: Replace the single `LLMTestCase` with a list generated from JSON/CSV fixtures to cover broader scenarios.
- **Introduce domain-specific metrics**: Create new `GEval` instances or custom metric classes to judge factuality, legal compliance, or brand tone.
- **Support additional providers**: Follow the `ClaudeModel` pattern to wrap other LLM endpoints while preserving the DeepEval interface.
- **Integrate structured outputs**: Modify `evaluation_params` to include JSON schemas or function-call traces when assessing tool-augmented agents.

## Troubleshooting Checklist

- **Authentication errors**: Confirm `ANTHROPIC_API_KEY` is exported or present in a local `.env` file.
- **Missing prompts**: Ensure filenames in `read_prompt` calls match the assets in `prompts/`; the helpers resolve relative to the project root.
- **Judge drift**: If `GEval` scores fluctuate, pin the judge model (`claude-3-5-haiku-20241022`) or snapshot completions.
- **High latency or cost**: Reduce `max_tokens`, preload judgments, or run smaller subsets during development.

By following this blueprint, you can plan, execute, and scale comprehensive LLM evaluations using DeepEval while keeping the workflow grounded in reproducible assets and actionable metrics.
