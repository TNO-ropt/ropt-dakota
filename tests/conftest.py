from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from ropt.workflow import create_evaluator

_Function = Callable[[NDArray[np.float64], int], float]


def pytest_addoption(parser: Any) -> Any:
    parser.addoption(
        "--run-external",
        action="store_true",
        default=False,
        help="run tests with external optimizers",
    )


def pytest_collection_modifyitems(config: Any, items: Sequence[Any]) -> None:
    if not config.getoption("--run-external"):
        skip_external = pytest.mark.skip(reason="need --run-external option to run")
        for item in items:
            if "external" in item.keywords:
                item.add_marker(skip_external)


def _compute_distance_squared(
    variables: NDArray[np.float64], _: int, target: NDArray[np.float64]
) -> float:
    return float(((variables - target) ** 2).sum())


@pytest.fixture(name="test_functions", scope="session")
def fixture_test_functions() -> tuple[_Function, _Function]:
    return (
        partial(_compute_distance_squared, target=np.array([0.5, 0.5, 0.5])),
        partial(_compute_distance_squared, target=np.array([-1.5, -1.5, 0.5])),
    )


def _function(
    variables: NDArray[np.float64],
    realization: int,
    _: int,
    /,
    *,
    test_functions: list[_Function],
) -> NDArray[np.float64]:
    return np.fromiter(
        (func(variables, realization) for func in test_functions), dtype=np.float64
    )


@pytest.fixture(scope="session")
def evaluator(test_functions: Any) -> Any:
    def _evaluator(test_functions: list[_Function] = test_functions) -> Any:
        return create_evaluator(
            "function_evaluator",
            function=partial(_function, test_functions=test_functions),
        )

    return _evaluator
