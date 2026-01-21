from deepeval import evaluate

from tests.utils.requirement_evaluation import (
    build_requirement_metrics,
    requirement_hyperparameters,
    load_requirement_test_cases,
)


test_cases = load_requirement_test_cases()

evaluate(
    test_cases,
    build_requirement_metrics(),
    hyperparameters=requirement_hyperparameters,
)
