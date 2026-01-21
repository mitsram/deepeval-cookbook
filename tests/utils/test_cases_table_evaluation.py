from typing import List

from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from tests.utils.requirement_evaluation import (
    build_hyperparameters,
    claude_model,
    load_dataset_test_cases,
)


def build_test_cases_table_metrics(model: DeepEvalBaseLLM | None = None) -> List[GEval]:
    """Configure GEval metrics for validating the test-cases table output."""

    metric_model = model or claude_model
    return [
        GEval(
            name="Table Format",
            criteria=(
                "Confirm the actual output is a well-formed markdown table with the required headings "
                "TC-ID, Description, Type, Expected Outcome, Reference."
            ),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.5,
            model=metric_model,
        ),
        GEval(
            name="Content Alignment",
            criteria=(
                "Check that the actual output covers both positive and negative scenarios using "
                "equivalence partitioning and boundary analysis details as described in the expected output."
            ),
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=0.5,
            model=metric_model,
        ),
        GEval(
            name="Completeness",
            criteria=(
                "Ensure each table row contains a filled TC-ID, Description, Type, Expected Outcome, and Reference."
            ),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.5,
            model=metric_model,
        ),
    ]


def load_test_cases_table() -> List[LLMTestCase]:
    """Load the password reset test-cases table dataset."""

    return load_dataset_test_cases("test_cases_table_output.json")


test_cases_table_hyperparameters = build_hyperparameters(
    "test_cases_table_output_v1",
    "prompts/test_cases_table_output.md",
)
