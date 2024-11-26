from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from ropt.enums import ConstraintType, EventType, OptimizerExitCode
from ropt.plan import BasicOptimizer, Event
from ropt.results import GradientResults

from ropt_dakota.dakota import _SUPPORTED_METHODS


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    return {
        "variables": {
            "initial_values": [0.0, 0.0, 0.1],
        },
        "optimizer": {
            "method": "dakota/default",
            "tolerance": 1e-6,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
        "gradient": {
            "perturbation_magnitudes": 0.01,
        },
    }


def test_dakota_unconstrained(enopt_config: Any, evaluator: Any) -> None:
    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)


def test_dakota_option(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["optimizer"]["options"] = ["max_iterations = 0"]
    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(
        variables,
        enopt_config["variables"]["initial_values"],
        atol=0.02,
    )


@pytest.mark.parametrize(
    # The conmin algorithms are not tested, since they produce some output to
    # the terminal that we are not able to suppress. The soga method crashes
    # occasionally.
    "method",
    sorted(_SUPPORTED_METHODS - {"conmin_mfd", "conmin_frcg", "soga"}),
)
def test_dakota_bound_constraint(
    enopt_config: Any, method: str, evaluator: Any
) -> None:
    enopt_config["optimizer"]["method"] = f"dakota/{method}"
    enopt_config["variables"]["lower_bounds"] = -1.0
    enopt_config["variables"]["upper_bounds"] = [1.0, 1.0, 0.2]
    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    # Some methods do not easily convert, we just test if the ran:
    if method not in ("coliny_ea", "moga"):
        assert np.allclose(variables, [0.0, 0.0, 0.2], atol=0.02)


def test_dakota_eq_linear_constraint(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 1]],
        "rhs_values": [1.0, 0.75],
        "types": [ConstraintType.EQ, ConstraintType.EQ],
    }
    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.25, 0.0, 0.75], atol=0.02)


def test_dakota_ge_linear_constraint(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["linear_constraints"] = {
        "coefficients": [[-1, 0, -1]],
        "rhs_values": -0.4,
        "types": [ConstraintType.GE],
    }
    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)


def test_dakota_le_linear_constraint(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1]],
        "rhs_values": 0.4,
        "types": [ConstraintType.LE],
    }
    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)


def test_dakota_le_ge_linear_constraints(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [-1, 0, -1]],
        "rhs_values": [0.4, -0.4],
        "types": [ConstraintType.LE, ConstraintType.GE],
    }
    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)


def test_dakota_eq_nonlinear_constraint(
    enopt_config: Any, evaluator: Any, test_functions: Any
) -> None:
    enopt_config["nonlinear_constraints"] = {
        "rhs_values": 1.0,
        "types": [ConstraintType.EQ],
    }
    test_functions = (
        *test_functions,
        lambda variables: cast(NDArray[np.float64], variables[0] + variables[2]),
    )
    variables = BasicOptimizer(enopt_config, evaluator(test_functions)).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.25, 0.0, 0.75], atol=0.02)


@pytest.mark.parametrize("bound_type", [ConstraintType.LE, ConstraintType.GE])
def test_dakota_ineq_nonlinear_constraint(
    enopt_config: Any,
    bound_type: ConstraintType,
    evaluator: Any,
    test_functions: Any,
) -> None:
    enopt_config["nonlinear_constraints"] = {
        "rhs_values": 0.4 if bound_type == ConstraintType.LE else -0.4,
        "types": [bound_type],
    }
    weight = 1.0 if bound_type == ConstraintType.LE else -1.0
    test_functions = (
        *test_functions,
        lambda variables: cast(
            NDArray[np.float64], weight * variables[0] + weight * variables[2]
        ),
    )
    variables = BasicOptimizer(enopt_config, evaluator(test_functions)).run().variables
    assert variables is not None
    assert np.allclose(variables, [-0.05, 0.0, 0.45], atol=0.02)


def test_dakota_failed_realizations(enopt_config: Any, evaluator: Any) -> None:
    def func_p(_0: NDArray[np.float64]) -> float:
        return 1.0

    def func_q(_0: NDArray[np.float64]) -> float:
        return np.nan

    functions = [func_p, func_q]

    assert (
        BasicOptimizer(
            enopt_config,
            evaluator(functions),
        )
        .run()
        .exit_code
        == OptimizerExitCode.TOO_FEW_REALIZATIONS
    )


def test_dakota_user_abort(enopt_config: Any, evaluator: Any) -> None:
    def observer(event: Event) -> None:
        assert event.results is not None
        if event.results[0].result_id == 2:
            plan.abort_optimization()

    plan = BasicOptimizer(enopt_config, evaluator()).add_observer(
        EventType.FINISHED_EVALUATION, observer
    )
    plan.run()
    assert plan.results is not None
    assert plan.results.result_id == 2
    assert plan.exit_code == OptimizerExitCode.USER_ABORT


def test_dakota_split_evaluations(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["optimizer"]["split_evaluations"] = True
    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)

    enopt_config["optimizer"]["split_evaluations"] = True
    enopt_config["optimizer"]["speculative"] = True
    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)


def test_dakota_optimizer_variables_subset(enopt_config: Any, evaluator: Any) -> None:
    enopt_config["variables"]["initial_values"] = [0.0, 0.0, 0.1]
    enopt_config["variables"]["lower_bounds"] = -1.0
    enopt_config["variables"]["upper_bounds"] = 1.0

    # Fix the second variables, the test function still has the same optimal
    # values for the other parameters:
    enopt_config["variables"]["indices"] = [0, 2]

    def assert_gradient(event: Event) -> None:
        assert event.results is not None
        for item in event.results:
            if isinstance(item, GradientResults):
                assert item.gradients is not None
                assert item.gradients.weighted_objective[1] == 0.0
                assert np.all(np.equal(item.gradients.objectives[:, 1], 0.0))

    variables = (
        BasicOptimizer(enopt_config, evaluator())
        .add_observer(EventType.FINISHED_EVALUATION, assert_gradient)
        .run()
        .variables
    )
    assert variables is not None
    assert np.allclose(variables, [0.0, 0.0, 0.5], atol=0.02)


def test_dakota_optimizer_variables_subset_linear_constraints(
    enopt_config: Any, evaluator: Any
) -> None:
    # Set the second variable a constant value, this will not affect the
    # optimization of the other variables in this particular test problem:
    enopt_config["variables"]["initial_values"] = [0.0, 1.0, 0.1]
    # The second and third constraints are dropped because they involve
    # variables that are not optimized. They are still checked by the monitor:
    enopt_config["linear_constraints"] = {
        "coefficients": [[1, 0, 1], [0, 1, 0], [1, 1, 1]],
        "rhs_values": [1.0, 1.0, 2.0],
        "types": [ConstraintType.EQ, ConstraintType.EQ, ConstraintType.EQ],
    }
    enopt_config["variables"]["indices"] = [0, 2]
    variables = BasicOptimizer(enopt_config, evaluator()).run().variables
    assert variables is not None
    assert np.allclose(variables, [0.25, 1.0, 0.75], atol=0.02)


def test_dakota_output_dir(tmp_path: Path, enopt_config: Any, evaluator: Any) -> None:
    output_dir = tmp_path / "outputdir"
    output_dir.mkdir()
    enopt_config["optimizer"]["output_dir"] = output_dir
    enopt_config["optimizer"]["max_functions"] = 1
    BasicOptimizer(enopt_config, evaluator()).run()
    assert (output_dir / "dakota").exists()
    BasicOptimizer(enopt_config, evaluator()).run()
    assert (output_dir / "dakota-001").exists()
    BasicOptimizer(enopt_config, evaluator()).run()
    assert (output_dir / "dakota-002").exists()
