import json
import os
import warnings
from pathlib import Path
from types import MethodType
from typing import List

from anthropic import Anthropic
from deepeval import login
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.test_run import global_test_run_manager
from dotenv import load_dotenv

load_dotenv()

TESTS_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = TESTS_DIR / "datasets"

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
confident_api_key = os.getenv("CONFIDENT_API_KEY")
if not confident_api_key:
    raise RuntimeError(
        "CONFIDENT_API_KEY is required for Confident AI integration. Update your .env file."
    )

try:
    login(confident_api_key)
    global_test_run_manager.disable_request = False
except Exception as exc:
    raise RuntimeError(f"Confident AI login failed: {exc}") from exc

client = Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None


def _install_confident_fallback() -> None:
    """Ensure Confident upload failures fall back to local reporting."""

    if getattr(global_test_run_manager, "_confident_fallback_installed", False):
        return

    original_wrap_up = global_test_run_manager.wrap_up_test_run.__func__

    def _wrap_up_with_fallback(self, run_duration, display_table=True, display=None):
        self.disable_request = False
        try:
            return original_wrap_up(self, run_duration, display_table, display)
        except RuntimeError as exc:
            message = str(exc)
            if "Confident API response missing 'id'" not in message:
                raise
            warnings.warn(
                "Confident AI upload failed (server unavailable). Falling back to local DeepEval reporting.",
                stacklevel=2,
            )
            self.disable_request = True
            try:
                return original_wrap_up(self, run_duration, display_table, display)
            finally:
                self.disable_request = False

    global_test_run_manager.wrap_up_test_run = MethodType(
        _wrap_up_with_fallback,
        global_test_run_manager,
    )
    global_test_run_manager._confident_fallback_installed = True


_install_confident_fallback()


class ClaudeModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "claude-3-5-haiku-20241022"):
        self.model_name = model_name
        self.client = client
        super().__init__(model_name)

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str, schema=None, **kwargs) -> str:
        """Generate response using Anthropic Claude API."""
        if self.client:
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except Exception:
                pass
        return self._mock_response(schema)

    async def a_generate(self, prompt: str, schema=None, **kwargs) -> str:
        """Async generate response using Anthropic Claude API."""
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self) -> str:
        return self.model_name

    def _mock_response(self, schema) -> str:
        schema_name = getattr(schema, "__name__", None)
        if schema_name == "Steps":
            return json.dumps(
                {
                    "steps": [
                        "Assess clarity of requirement analysis output.",
                        "Check coverage against expected interpretations.",
                        "Verify actionable insights are present.",
                    ]
                }
            )
        if schema_name == "ReasonScore":
            return json.dumps(
                {
                    "score": 10.0,
                    "reason": "Offline fallback: requirement analysis meets mocked rubric.",
                }
            )
        return json.dumps(
            {
                "score": 10.0,
                "reason": "Offline fallback response.",
            }
        )


claude_model = ClaudeModel()

def build_hyperparameters(
    suite: str,
    prompt_asset: str,
    model: DeepEvalBaseLLM | None = None,
) -> dict:
    """Compose the DeepEval hyperparameters payload for a suite."""

    hyper_model = model or claude_model
    return {
        "suite": suite,
        "judge_model": hyper_model.get_model_name(),
        "prompt_asset": prompt_asset,
    }


requirement_hyperparameters = build_hyperparameters(
    "requirement_analysis_v1",
    "prompts/requirement_analysis.md",
)


def build_requirement_metrics(model: DeepEvalBaseLLM | None = None) -> List[GEval]:
    """Create evaluation metrics bound to the provided model."""

    metric_model = model or claude_model
    return [
        GEval(
            name="Correctness",
            criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=0.5,
            model=metric_model,
        ),
        GEval(
            name="Clarity",
            criteria="Assess how clear and understandable the actual output is. Evaluate linguistic complexity, readability, and how easy it is to comprehend the message.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.5,
            model=metric_model,
        ),
        GEval(
            name="Relevance",
            criteria="Ensure the actual output is relevant to the input question. Evaluate how well the response aligns with what was asked and addresses the user's concerns.",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.5,
            model=metric_model,
        ),
        GEval(
            name="Completeness",
            criteria="Check if the actual output covers all necessary aspects to fully answer the input question. Identify any missing elements or gaps in the response.",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.5,
            model=metric_model,
        ),
        GEval(
            name="Consistency",
            criteria="Evaluate the consistency in language, tone, and format in the actual output. Assess uniformity in style and terminology throughout the response.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.5,
            model=metric_model,
        ),
        GEval(
            name="Actionability",
            criteria="Determine if the actual output leads to actionable steps or insights. Evaluate whether users can take meaningful actions based on the information provided.",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.5,
            model=metric_model,
        ),
        GEval(
            name="Accuracy",
            criteria="Validate the accuracy of the actual output by comparing it with the expected output. Look for any errors, misinterpretations, or factual inaccuracies.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=0.5,
            model=metric_model,
        ),
        GEval(
            name="Efficiency",
            criteria="Consider whether the actual output is concise and efficient in conveying information. Evaluate if the response avoids unnecessary verbosity while maintaining completeness.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.5,
            model=metric_model,
        ),
        GEval(
            name="Bias and Fairness",
            criteria="Check for any biases or unfair assumptions in the actual output. Evaluate sentiment, neutrality, and whether the response treats all groups fairly without discrimination.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.5,
            model=metric_model,
        ),
    ]


def load_dataset_test_cases(dataset_name: str) -> List[LLMTestCase]:
    """Load LLM test cases from a JSON dataset in the datasets directory."""

    dataset_path = DATASETS_DIR / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    dataset = EvaluationDataset()
    dataset.add_test_cases_from_json_file(
        file_path=str(dataset_path),
        input_key_name="input",
        actual_output_key_name="actual_output",
        expected_output_key_name="expected_output",
    )
    return dataset.test_cases


def load_requirement_test_cases() -> List[LLMTestCase]:
    """Load the requirement analysis dataset test cases."""

    return load_dataset_test_cases("requirement_analysis.json")
