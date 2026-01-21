from deepeval import evaluate

from tests.utils.test_cases_table_evaluation import (
    build_test_cases_table_metrics,
    load_test_cases_table,
    test_cases_table_hyperparameters,
)


test_cases = load_test_cases_table()

evaluate(
    test_cases,
    build_test_cases_table_metrics(),
    hyperparameters=test_cases_table_hyperparameters,
)
