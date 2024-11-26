from functools import partial
from typing import Any, Callable

import numpy as np
import pytest
from numpy.typing import NDArray
from ropt.evaluator import Evaluator, EvaluatorContext, EvaluatorResult

_Function = Callable[[NDArray[np.float64]], float]


def _function_runner(
    variables: NDArray[np.float64],
    metadata: EvaluatorContext,
    functions: list[_Function],
) -> EvaluatorResult:
    objective_count = metadata.config.objectives.weights.size
    constraint_count = (
        0
        if metadata.config.nonlinear_constraints is None
        else metadata.config.nonlinear_constraints.rhs_values.size
    )
    objective_results = np.zeros(
        (variables.shape[0], objective_count), dtype=np.float64
    )
    constraint_results = (
        np.zeros((variables.shape[0], constraint_count), dtype=np.float64)
        if constraint_count > 0
        else None
    )
    for sim, realization in enumerate(metadata.realizations):
        for idx in range(objective_count):
            if (
                metadata.active_objectives is None
                or metadata.active_objectives[idx, realization]
            ):
                function = functions[idx]
                objective_results[sim, idx] = function(variables[sim, :])
        for idx in range(constraint_count):
            if (
                metadata.active_constraints is None
                or metadata.active_constraints[idx, realization]
            ):
                function = functions[idx + objective_count]
                assert constraint_results is not None
                constraint_results[sim, idx] = function(variables[sim, :])
    return EvaluatorResult(
        objectives=objective_results,
        constraints=constraint_results,
    )


def _compute_distance_squared(
    variables: NDArray[np.float64], target: NDArray[np.float64]
) -> float:
    return float(((variables - target) ** 2).sum())


@pytest.fixture(name="test_functions", scope="session")
def fixture_test_functions() -> tuple[_Function, _Function]:
    return (
        partial(_compute_distance_squared, target=np.array([0.5, 0.5, 0.5])),
        partial(_compute_distance_squared, target=np.array([-1.5, -1.5, 0.5])),
    )


@pytest.fixture(scope="session")
def evaluator(test_functions: Any) -> Callable[[list[_Function]], Evaluator]:
    def _evaluator(test_functions: list[_Function] = test_functions) -> Evaluator:
        return partial(_function_runner, functions=test_functions)

    return _evaluator
