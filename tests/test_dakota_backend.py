# ruff: noqa: RUF069

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError
from ropt.enums import ExitCode
from ropt.results import GradientResults, Results
from ropt.workflow import BasicOptimizer, validate_backend_options

from ropt_dakota.dakota import _SUPPORTED_METHODS

initial_values = [0.0, 0.0, 0.1]


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "variable_count": len(initial_values),
            "perturbation_magnitudes": 0.01,
        },
        "backend": {
            "method": "dakota/default",
            "convergence_tolerance": 1e-6,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
    }


def test_dakota_invalid_options(config: Any) -> None:
    config["backend"]["method"] = "optpp_q_newton"
    config["backend"]["options"] = [
        "max_iterations = 0",
        "merit_function el_bakry",
    ]
    validate_backend_options("optpp_q_newton", config["backend"]["options"])

    config["backend"]["options"] = [
        "max_iterations = 0",
        "search_method = 1",
        "merit_function el_bakry",
    ]
    with pytest.raises(
        ValidationError, match=r"Input should be 'value_based_line_search',"
    ):
        validate_backend_options("optpp_q_newton", config["backend"]["options"])

    config["backend"]["options"] = [
        "max_iterations = 0",
        "foo = xyz",
        "bar",
        "merit_function el_bakry",
    ]
    with pytest.raises(
        ValidationError, match=r"Unknown or unsupported option\(s\): `foo`, `bar`"
    ):
        validate_backend_options("optpp_q_newton", config["backend"]["options"])


@pytest.mark.parametrize(
    "external", ["", pytest.param("external/", marks=pytest.mark.external)]
)
def test_dakota_unconstrained(config: Any, evaluator: Any, external: str) -> None:
    config["backend"]["method"] = f"{external}optpp_q_newton"
    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )


@pytest.mark.parametrize(
    # The conmin algorithms are not tested, since they produce some output to
    # the terminal that we are not able to suppress. The soga method crashes
    # occasionally.
    "method",
    sorted(_SUPPORTED_METHODS - {"conmin_mfd", "conmin_frcg", "soga"}),
)
def test_dakota_bound_constraint(config: Any, method: str, evaluator: Any) -> None:
    config["backend"]["method"] = f"dakota/{method}"
    config["variables"]["lower_bounds"] = -1.0
    config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    # Some methods do not easily convert, we just test if the ran:
    if method not in {"coliny_ea", "moga"}:
        assert np.allclose(
            optimizer.results.evaluations.variables, [0.0, 0.0, 0.2], atol=0.02
        )


def test_dakota_eq_linear_constraint(config: Any, evaluator: Any) -> None:
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 1]],
        "lower_bounds": [1.0, 0.75],
        "upper_bounds": [1.0, 0.75],
    }
    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.25, 0.0, 0.75], atol=0.02
    )


def test_dakota_ge_linear_constraint(config: Any, evaluator: Any) -> None:
    config["linear_constraints"] = {
        "coefficients": [[-1, 0, -1]],
        "lower_bounds": -0.4,
        "upper_bounds": np.inf,
    }
    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02
    )


def test_dakota_le_linear_constraint(config: Any, evaluator: Any) -> None:
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": -np.inf,
        "upper_bounds": 0.4,
    }
    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02
    )


def test_dakota_le_ge_linear_constraints(config: Any, evaluator: Any) -> None:
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [-1, 0, -1]],
        "lower_bounds": [-np.inf, -0.4],
        "upper_bounds": [0.4, np.inf],
    }
    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02
    )


def test_dakota_le_ge_linear_constraints_two_sided(config: Any, evaluator: Any) -> None:
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [1, 0, 1]],
        "lower_bounds": [-np.inf, 0.0],
        "upper_bounds": [0.3, np.inf],
    }

    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.1, 0.0, 0.4], atol=0.02
    )

    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "lower_bounds": [0.0],
        "upper_bounds": [0.3],
    }

    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.1, 0.0, 0.4], atol=0.02
    )


def test_dakota_eq_nonlinear_constraint(
    config: Any, evaluator: Any, test_functions: Any
) -> None:
    config["nonlinear_constraints"] = {
        "lower_bounds": 1.0,
        "upper_bounds": 1.0,
    }
    test_functions = (
        *test_functions,
        lambda variables, _: variables[0] + variables[2],
    )
    optimizer = BasicOptimizer(config, evaluator(test_functions))
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.25, 0.0, 0.75], atol=0.02
    )


@pytest.mark.parametrize(
    ("lower_bounds", "upper_bounds"), [(-np.inf, 0.4), (-0.4, np.inf)]
)
def test_dakota_ineq_nonlinear_constraint(
    config: Any,
    lower_bounds: Any,
    upper_bounds: Any,
    evaluator: Any,
    test_functions: Any,
) -> None:
    config["nonlinear_constraints"] = {
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }
    weight = 1.0 if upper_bounds == 0.4 else -1.0
    test_functions = (
        *test_functions,
        lambda variables, _: weight * variables[0] + weight * variables[2],
    )
    optimizer = BasicOptimizer(config, evaluator(test_functions))
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.05, 0.0, 0.45], atol=0.02
    )


def test_dakota_ineq_nonlinear_constraints_two_sided(
    config: Any,
    evaluator: Any,
    test_functions: Any,
) -> None:
    config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    config["nonlinear_constraints"] = {
        "lower_bounds": [0.01, 0.0],
        "upper_bounds": [0.01, 0.3],
    }
    test_functions = (
        *test_functions,
        lambda variables, _: variables[1],
        lambda variables, _: variables[0] + variables[2],
    )

    optimizer = BasicOptimizer(config, evaluator(test_functions))
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.1, 0.01, 0.4], atol=0.02
    )


def test_dakota_ineq_nonlinear_constraints_eq_ineq(
    config: Any,
    evaluator: Any,
    test_functions: Any,
) -> None:
    config["variables"]["lower_bounds"] = [-1.0, -1.0, -1.0]
    config["variables"]["upper_bounds"] = [1.0, 1.0, 1.0]
    config["nonlinear_constraints"] = {
        "lower_bounds": [0.01, 0.0],
        "upper_bounds": [0.01, 0.3],
    }
    test_functions = (
        *test_functions,
        lambda variables, _: variables[1],
        lambda variables, _: variables[0] + variables[2],
    )

    optimizer = BasicOptimizer(config, evaluator(test_functions))
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [-0.1, 0.01, 0.4], atol=0.02
    )


def test_dakota_failed_realizations(config: Any, evaluator: Any) -> None:
    def func_p(_0: NDArray[np.float64], _1: int) -> float:
        return 1.0

    def func_q(_0: NDArray[np.float64], _1: int) -> float:
        return np.nan

    functions = [func_p, func_q]

    optimizer = BasicOptimizer(
        config,
        evaluator(functions),
    )
    exit_code = optimizer.run(initial_values)
    assert exit_code == ExitCode.TOO_FEW_REALIZATIONS


def test_dakota_user_abort(config: Any, evaluator: Any) -> None:
    last_evaluation = 0

    def _abort() -> bool:
        nonlocal last_evaluation

        if last_evaluation == 2:
            return True
        last_evaluation += 1
        return False

    optimizer = BasicOptimizer(config, evaluator())
    optimizer.set_abort_callback(_abort)
    exit_code = optimizer.run(initial_values)
    assert optimizer.results is not None
    assert last_evaluation == 2
    assert exit_code == ExitCode.USER_ABORT


def test_dakota_evaluation_policy_separate(config: Any, evaluator: Any) -> None:
    config["gradient"] = {"evaluation_policy": "separate"}
    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )

    config["gradient"] = {"evaluation_policy": "separate"}
    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )


def test_dakota_optimizer_variables_subset(config: Any, evaluator: Any) -> None:
    config["variables"]["lower_bounds"] = -1.0
    config["variables"]["upper_bounds"] = 1.0

    # Fix the second variables, the test function still has the same optimal
    # values for the other parameters:
    config["variables"]["mask"] = [True, False, True]

    def assert_gradient(results: tuple[Results, ...]) -> None:
        for item in results:
            if isinstance(item, GradientResults):
                assert item.gradients is not None
                assert item.gradients.target_objective[1] == 0.0
                assert np.all(np.equal(item.gradients.objectives[:, 1], 0.0))

    optimizer = BasicOptimizer(config, evaluator())
    optimizer.set_results_callback(assert_gradient)
    optimizer.run(initial_values)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.0, 0.0, 0.5], atol=0.02
    )


def test_dakota_optimizer_variables_subset_linear_constraints(
    config: Any, evaluator: Any
) -> None:
    # Set the second variable a constant value, this will not affect the
    # optimization of the other variables in this particular test problem: The
    # second and third constraints are dropped because they involve variables
    # that are not optimized. They are still checked by the monitor:
    config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 0], [1, 1, 1]],
        "lower_bounds": [1.0, 1.0, 2.0],
        "upper_bounds": [1.0, 1.0, 2.0],
    }
    config["variables"]["mask"] = [True, False, True]
    initial = initial_values.copy()
    initial[1] = 1.0
    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial)
    assert optimizer.results is not None
    assert np.allclose(
        optimizer.results.evaluations.variables, [0.25, 1.0, 0.75], atol=0.02
    )


def test_dakota_output_dir(tmp_path: Path, config: Any, evaluator: Any) -> None:
    output_dir = tmp_path / "outputdir"
    output_dir.mkdir()
    config["optimizer"] = {"max_functions": 1, "output_dir": output_dir}
    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert (output_dir / "dakota").exists()
    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert (output_dir / "dakota-001").exists()
    optimizer = BasicOptimizer(config, evaluator())
    optimizer.run(initial_values)
    assert (output_dir / "dakota-002").exists()
