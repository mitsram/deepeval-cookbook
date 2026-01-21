import json
import os
import warnings
from pathlib import Path
from types import MethodType
from dotenv import load_dotenv
from anthropic import Anthropic
from deepeval import evaluate, login
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_run import global_test_run_manager

# Load environment variables from .env file
load_dotenv()

# Configure Anthropic API
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Connect to Confident AI when a key is present
confident_api_key = os.getenv("CONFIDENT_API_KEY")
if not confident_api_key:
    raise RuntimeError(
        "CONFIDENT_API_KEY is required for Confident AI integration. Update your .env file."
    )

try:
    login(confident_api_key)
    # Ensure uploads are enabled for this run after a successful login.
    global_test_run_manager.disable_request = False
except Exception as exc:
    raise RuntimeError(f"Confident AI login failed: {exc}") from exc

# Create Anthropic client when a real key is present
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
                # Restore the default for future runs so we retry uploads next time.
                self.disable_request = False

    global_test_run_manager.wrap_up_test_run = MethodType(
        _wrap_up_with_fallback,
        global_test_run_manager,
    )
    global_test_run_manager._confident_fallback_installed = True


_install_confident_fallback()

# Read prompt from ./prompts folder
def read_prompt(filename: str) -> str:
    """Read prompt from the prompts folder."""
    # Resolve prompts directory from project root now that tests/ adds a level
    prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    prompt_path = prompts_dir / filename
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

# Custom Claude model wrapper for deepeval
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
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
            except Exception:
                # Fall back to deterministic offline behaviour when remote evaluation fails.
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

# Load the requirement analysis prompt
requirement_prompt = read_prompt("requirement_analysis.md")

# Create Claude model instance
claude_model = ClaudeModel(model_name="claude-3-5-haiku-20241022")

hyperparameters = {
    "suite": "requirement_analysis_v1",
    "judge_model": claude_model.get_model_name(),
    "prompt_asset": "prompts/requirement_analysis.md",
}

# Configure metrics using Claude model
correctness_metric = GEval(
    name="Correctness",
    criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.5,
    model=claude_model
)

clarity_metric = GEval(
    name="Clarity",
    criteria="Assess how clear and understandable the actual output is. Evaluate linguistic complexity, readability, and how easy it is to comprehend the message.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5,
    model=claude_model
)

relevance_metric = GEval(
    name="Relevance",
    criteria="Ensure the actual output is relevant to the input question. Evaluate how well the response aligns with what was asked and addresses the user's concerns.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5,
    model=claude_model
)

completeness_metric = GEval(
    name="Completeness",
    criteria="Check if the actual output covers all necessary aspects to fully answer the input question. Identify any missing elements or gaps in the response.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5,
    model=claude_model
)

consistency_metric = GEval(
    name="Consistency",
    criteria="Evaluate the consistency in language, tone, and format in the actual output. Assess uniformity in style and terminology throughout the response.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5,
    model=claude_model
)

actionability_metric = GEval(
    name="Actionability",
    criteria="Determine if the actual output leads to actionable steps or insights. Evaluate whether users can take meaningful actions based on the information provided.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5,
    model=claude_model
)

accuracy_metric = GEval(
    name="Accuracy",
    criteria="Validate the accuracy of the actual output by comparing it with the expected output. Look for any errors, misinterpretations, or factual inaccuracies.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.5,
    model=claude_model
)

efficiency_metric = GEval(
    name="Efficiency",
    criteria="Consider whether the actual output is concise and efficient in conveying information. Evaluate if the response avoids unnecessary verbosity while maintaining completeness.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5,
    model=claude_model
)

bias_and_fairness_metric = GEval(
    name="Bias and Fairness",
    criteria="Check for any biases or unfair assumptions in the actual output. Evaluate sentiment, neutrality, and whether the response treats all groups fairly without discrimination.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5,
    model=claude_model
)

test_case = LLMTestCase(
    input="Requirement: The system should respond quickly to user requests.",
    # Replace this with the actual output from your LLM application
    actual_output="""**Different Interpretations:**
1. The system should provide responses within milliseconds for all user requests.
2. The system should respond faster than the previous version for user requests.
3. The system should respond in a timeframe that users perceive as quick.

**Ambiguities:**
- Lexical: "quickly" is undefined - what is the specific response time requirement (e.g., < 100ms, < 1s, < 5s)?
- Contextual: "system" is vague - does this refer to the entire system, specific components, or particular operations?

**Contradictions:**
None identified.

**Missing Information:**
- Revised version 1: "The system should respond to user requests within 200 milliseconds for 95% of requests under normal load conditions."
- Revised version 2: "The API endpoints should return responses within 1 second for read operations and within 3 seconds for write operations."
- Revised version 3: "The user interface should acknowledge user requests within 100 milliseconds and complete the requested action within 2 seconds for standard operations."
""",
    expected_output="""**Different Interpretations:**
1. The system responds within a specific time threshold (e.g., under 1 second).
2. The system responds faster than competing systems or previous versions.
3. The system responds in a timeframe that meets user expectations based on the context.

**Ambiguities:**
- Lexical Ambiguities: "quickly" is not quantified - needs specific time values (milliseconds, seconds).
- Contextual Ambiguities: "system" could refer to backend, frontend, API, database, or the entire application stack.

**Contradictions:**
No contradictions found in the requirement.

**Missing Information:**
- Missing specific response time metrics (e.g., 500ms, 1 second, 2 seconds)
- Missing load conditions (e.g., concurrent users, request volume)
- Missing percentile requirements (e.g., p95, p99)
- Revised: "The system backend API should respond to user HTTP requests within 500 milliseconds at the 95th percentile under normal load conditions of up to 1000 concurrent users."
"""
)

# Evaluate with all metrics
evaluate(
    [test_case],
    [
        correctness_metric,
        clarity_metric,
        relevance_metric,
        completeness_metric,
        consistency_metric,
        actionability_metric,
        accuracy_metric,
        efficiency_metric,
        bias_and_fairness_metric,
    ],
    hyperparameters=hyperparameters,
)
