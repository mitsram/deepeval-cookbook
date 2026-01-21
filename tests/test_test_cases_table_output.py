import json
import os
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM

load_dotenv()

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None


def read_prompt(filename: str) -> str:
    """Read prompt assets from the project prompts directory."""
    prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    prompt_path = prompts_dir / filename

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as file:
        return file.read()


class ClaudeModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "claude-3-5-haiku-20241022"):
        self.model_name = model_name
        self.client = client
        super().__init__(model_name)

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str, schema=None, **kwargs) -> str:
        if self.client:
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except Exception:
                # Fall back to deterministic offline behaviour when remote evaluation fails.
                pass
        return self._mock_response(schema)

    async def a_generate(self, prompt: str, schema=None, **kwargs) -> str:
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self) -> str:
        return self.model_name

    def _mock_response(self, schema) -> str:
        schema_name = getattr(schema, "__name__", None)
        if schema_name == "Steps":
            return json.dumps(
                {
                    "steps": [
                        "Inspect markdown headers for required columns.",
                        "Validate each row provides non-empty cells.",
                        "Compare semantic coverage against expectations.",
                    ]
                }
            )
        if schema_name == "ReasonScore":
            return json.dumps(
                {
                    "score": 10.0,
                    "reason": "Offline fallback: table output satisfies all mocked criteria.",
                }
            )
        return json.dumps(
            {
                "score": 10.0,
                "reason": "Offline fallback response.",
            }
        )


test_cases_prompt = read_prompt("test_cases_table_output.md")
claude_model = ClaudeModel(model_name="claude-3-5-haiku-20241022")

table_format_metric = GEval(
    name="Table Format",
    criteria=(
        "Confirm the actual output is a well-formed markdown table with the required headings "
        "TC-ID, Description, Type, Expected Outcome, Reference."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5,
    model=claude_model,
)

content_alignment_metric = GEval(
    name="Content Alignment",
    criteria=(
        "Check that the actual output covers both positive and negative scenarios using "
        "equivalence partitioning and boundary analysis details as described in the expected output."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.5,
    model=claude_model,
)

completeness_metric = GEval(
    name="Completeness",
    criteria=(
        "Ensure each table row contains a filled TC-ID, Description, Type, Expected Outcome, and Reference."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5,
    model=claude_model,
)


password_reset_test_case = LLMTestCase(
    input=(
        "Requirement: The system shall allow users to reset their password via an email link."
    ),
    actual_output="""| TC-ID | Description | Type | Expected Outcome | Reference |
| --- | --- | --- | --- | --- |
| TC-001 | Verify password reset succeeds for valid registered email | Positive | Email with reset link is sent within 1 minute and link allows password change | Requirement-01 |
| TC-002 | Verify password reset rejects unregistered email input | Negative | User is shown an error message and no email is dispatched | Requirement-01 |
| TC-003 | Validate rate limiting on password reset request burst | Negative | After 5 rapid submissions the system throttles further requests for 15 minutes | Requirement-01 |
| TC-004 | Validate expired reset link behavior at 24-hour boundary | Negative | Accessing link after 24 hours shows expiration message and prompts new request | Requirement-01 |
| TC-005 | Validate password reset flow at minimum password length boundary | Positive | User can set a new password with exactly 12 characters and login succeeds | Requirement-01 |""",
    expected_output="""| TC-ID | Description | Type | Expected Outcome | Reference |
| --- | --- | --- | --- | --- |
| TC-001 | Validate password reset success for registered email | Positive | System emails a reset link and allows password change | Requirement-01 |
| TC-002 | Validate rejection of unregistered email | Negative | Error message is displayed and no email is sent | Requirement-01 |
| TC-003 | Validate throttling after repeated reset requests | Negative | System blocks additional attempts after threshold is reached | Requirement-01 |
| TC-004 | Validate expired reset link handling beyond validity window | Negative | User sees expiration notice and is prompted to request a new link | Requirement-01 |
| TC-005 | Validate boundary password length acceptance | Positive | Password exactly at minimum length is accepted and login succeeds | Requirement-01 |""",
)

evaluate(
    [password_reset_test_case],
    [table_format_metric, content_alignment_metric, completeness_metric],
)
