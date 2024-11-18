"""This module implements the Dakota optimization plugin."""

from math import isfinite
from os import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Final

import numpy as np
from dakota import _USER_DATA, DakotaBase, DakotaInput, run_dakota
from numpy.typing import NDArray
from ropt.config.enopt import EnOptConfig, OptimizerConfig
from ropt.enums import ConstraintType
from ropt.exceptions import ConfigError
from ropt.plugins.optimizer.base import Optimizer, OptimizerCallback, OptimizerPlugin
from ropt.plugins.optimizer.utils import create_output_path, filter_linear_constraints

_PRECISION: Final[int] = 8
_LARGE_NUMBER_FOR_INF: Final = 1e30

_ConstraintIndices = tuple[
    NDArray[np.intc],
    NDArray[np.intc],
    NDArray[np.intc],
]

_SUPPORTED_METHODS: Final = {
    "optpp_q_newton",
    "conmin_mfd",
    "conmin_frcg",
    "mesh_adaptive_search",
    "coliny_ea",
    "soga",
    "moga",
    "asynch_pattern_search",
}


class DakotaOptimizer(Optimizer):
    """Plugin class for optimization via Dakota."""

    def __init__(
        self, config: EnOptConfig, optimizer_callback: OptimizerCallback
    ) -> None:
        """Initialize the optimizer implemented by the Dakota plugin.

        See the [ropt.plugins.optimizer.base.Optimizer][] abstract base class.

        # noqa
        """
        self._config = config
        self._optimizer_callback = optimizer_callback
        self._constraint_indices = self._get_constraint_indices()
        self._output_dir: Path

        _, _, self._method = self._config.optimizer.method.lower().rpartition("/")
        if self._method == "default":
            self._method = "optpp_q_newton"
        if self._method not in _SUPPORTED_METHODS:
            msg = f"Dakota optimizer algorithm {self._method} is not supported"
            raise NotImplementedError(msg)

    def start(self, initial_values: NDArray[np.float64]) -> None:
        """Start the optimization.

        See the [ropt.plugins.optimizer.base.Optimizer][] abstract base class.

        # noqa
        """
        if self._config.optimizer.output_dir is None:
            with TemporaryDirectory() as output_dir:
                self._output_dir = Path(output_dir)
                self._start(initial_values)
        else:
            self._output_dir = self._config.optimizer.output_dir
            self._start(initial_values)

    @property
    def allow_nan(self) -> bool:
        """Whether NaN is allowed.

        See the [ropt.plugins.optimizer.base.Optimizer][] abstract base class.

        # noqa
        """
        return False

    @property
    def is_parallel(self) -> bool:
        """Whether the current run is parallel.

        See the [ropt.plugins.optimizer.base.Optimizer][] abstract base class.

        # noqa
        """
        return False

    def _get_constraint_indices(self) -> _ConstraintIndices | None:
        if self._config.nonlinear_constraints is None:
            return None
        types = self._config.nonlinear_constraints.types
        return (
            np.fromiter(
                (idx for idx, type_ in enumerate(types) if type_ == ConstraintType.LE),
                dtype=np.intc,
            ),
            np.fromiter(
                (idx for idx, type_ in enumerate(types) if type_ == ConstraintType.GE),
                dtype=np.intc,
            ),
            np.fromiter(
                (idx for idx, type_ in enumerate(types) if type_ == ConstraintType.EQ),
                dtype=np.intc,
            ),
        )

    def _get_inputs(self, initial_values: NDArray[np.float64]) -> dict[str, list[str]]:
        return {
            "environment": [
                "tabular_graphics_data",
                "output_precision = 8",
            ],
            "method": self._get_method_section(),
            "model": ["single"],
            "variables": (
                self._get_variables_section(initial_values)
                + self._get_linear_constraints_section()
            ),
            "responses": [
                "num_objective_functions = 1",
                "analytic_gradients",
                "no_hessians",
                *self._get_responses_section(),
            ],
        }

    def _get_method_section(self) -> list[str]:
        inputs: list[str] = []
        inputs.append(self._method)
        # Scaling is always on
        inputs.append("scaling")
        iterations = self._config.optimizer.max_iterations
        if iterations is not None and self._method != "asynch_pattern_search":
            inputs.append(f"max_iterations = {iterations}")
        convergence_tolerance = self._config.optimizer.tolerance
        if convergence_tolerance is not None:
            tolerance_option = (
                "variable_tolerance"
                if self._method in ["mesh_adaptive_search", "asynch_pattern_search"]
                else "convergence_tolerance"
            )
            inputs.append(f"{tolerance_option} = {convergence_tolerance}")
        # The Dakota interface seems not be able to deal with with speculative
        # and split_evaluations simultaneously. Do not enable speculative if
        # split_evaluations is defined.
        if (
            self._config.optimizer.speculative
            and not self._config.optimizer.split_evaluations
        ):
            inputs.append("speculative")
        # Options are put in the method section:
        return inputs + self._get_options(self._method)

    def _get_options(self, algorithm: str) -> list[str]:
        inputs: list[str] = []
        if isinstance(self._config.optimizer.options, list):
            try:
                for option in self._config.optimizer.options:
                    stripped = option.strip()
                    if stripped.startswith("input_file"):
                        continue
                    if algorithm in [
                        "conmin_mfd",
                        "conmin_frcg",
                    ] and stripped.startswith("constraint_tolerance"):
                        continue
                    inputs.append(f"{option}")
            except TypeError as exc:
                msg = "Cannot parse Dakota optimization options"
                raise ConfigError(msg) from exc
        return inputs

    def _get_variables_section(self, initial_values: NDArray[np.float64]) -> list[str]:
        inputs: list[str] = []
        names = self._config.variables.names
        if names is None:
            names = tuple(f"variable{idx}" for idx in range(initial_values.size))
        lower_bounds = self._config.variables.lower_bounds
        upper_bounds = self._config.variables.upper_bounds
        variable_indices = self._config.variables.indices
        if variable_indices is not None:
            initial_values = initial_values[variable_indices]
            names = tuple(names[idx] for idx in variable_indices)
            lower_bounds = lower_bounds[variable_indices]
            upper_bounds = upper_bounds[variable_indices]
        inputs.append(f"continuous_design = {initial_values.size}")
        inputs.append(
            "initial_point "
            + " ".join(
                f"{initial_value:{_PRECISION}f}" for initial_value in initial_values
            ),
        )
        inputs.append(
            "lower_bounds "
            + " ".join(
                f"{bound:{_PRECISION}f}"
                if isfinite(bound)
                else f"-{_LARGE_NUMBER_FOR_INF}"
                for bound in lower_bounds
            ),
        )
        inputs.append(
            "upper_bounds "
            + " ".join(
                f"{bound:{_PRECISION}f}"
                if isfinite(bound)
                else f"+{_LARGE_NUMBER_FOR_INF}"
                for bound in upper_bounds
            ),
        )
        variable_names = [
            "_".join(str(name) for name in variable_name)
            if isinstance(variable_name, tuple)
            else str(variable_name)
            for variable_name in names
        ]
        inputs.append(
            "descriptors " + "  ".join(f"'{name}'" for name in variable_names),
        )
        return inputs

    def _get_linear_constraints(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.ubyte]]:
        linear_constraints_config = self._config.linear_constraints
        assert linear_constraints_config is not None

        if self._config.variables.indices is not None:
            linear_constraints_config = filter_linear_constraints(
                linear_constraints_config, self._config.variables.indices
            )

        coefficients = linear_constraints_config.coefficients
        rhs_values = linear_constraints_config.rhs_values
        types = linear_constraints_config.types

        return coefficients, rhs_values, types

    def _get_linear_constraints_section(self) -> list[str]:
        inputs: list[str] = []

        if self._config.linear_constraints is not None:
            all_coefficients, all_rhs_values, all_types = self._get_linear_constraints()

            coefficients = all_coefficients[all_types != ConstraintType.EQ, :]
            rhs_values = all_rhs_values[all_types != ConstraintType.EQ]
            rhs_values[rhs_values < -_LARGE_NUMBER_FOR_INF] = -_LARGE_NUMBER_FOR_INF
            rhs_values[rhs_values > _LARGE_NUMBER_FOR_INF] = _LARGE_NUMBER_FOR_INF
            types = all_types[all_types != ConstraintType.EQ]
            if types.size > 0:
                # Inequality linear constraints are defined as one-sided with a lower
                # bound. Earlier versions of this driver used upper bounds, and we stick
                # to that by using the negatives of the matrix and bound values for ge
                # constraints.
                coefficients[types == ConstraintType.GE, :] = -coefficients[
                    types == ConstraintType.GE, :
                ]
                rhs_values[types == ConstraintType.GE] = -rhs_values[
                    types == ConstraintType.GE
                ]
                # Add 0.0 to prevent -0.0 values:
                coefficients += 0.0
                rhs_values += 0.0
                inputs.append(
                    "linear_inequality_constraint_matrix = "
                    + "\n".join(
                        " ".join(
                            f"{value:{_PRECISION}f}" for value in coefficients[idx, :]
                        )
                        for idx in range(types.size)
                    ),
                )
                inputs.append(
                    "linear_inequality_lower_bounds = "
                    + " ".join(["-1e+30"] * types.size),
                )
                inputs.append(
                    "linear_inequality_upper_bounds = "
                    + " ".join(f"{value:{_PRECISION}f}" for value in rhs_values),
                )

            coefficients = all_coefficients[all_types == ConstraintType.EQ, :]
            rhs_values = all_rhs_values[all_types == ConstraintType.EQ]
            types = all_types[all_types == ConstraintType.EQ]
            if types.size > 0:
                inputs.append(
                    "linear_equality_constraint_matrix = "
                    + "\n".join(
                        " ".join(
                            f"{value:{_PRECISION}f}" for value in coefficients[idx, :]
                        )
                        for idx in range(types.size)
                    ),
                )
                inputs.append(
                    "linear_equality_targets ="
                    + " ".join(f"{value:{_PRECISION}f}" for value in rhs_values),
                )

        return inputs

    def _get_responses_section(self) -> list[str]:
        inputs: list[str] = []
        # ropt currently only allows for a single objective function
        inputs.append("objective_function_scale_types 'value'")
        inputs.append("objective_function_scales = 1.0")
        if self._constraint_indices is not None:
            ge_indices, le_indices, eq_indices = self._constraint_indices
            ineq_count = ge_indices.size + le_indices.size
            if ineq_count > 0:
                inputs.append(f"nonlinear_inequality_constraints = {ineq_count} ")
                inputs.append("nonlinear_inequality_upper_bounds" + " 0.0" * ineq_count)
                inputs.append("nonlinear_inequality_scale_types 'value'")
                inputs.append("nonlinear_inequality_scales" + " 1.0" * ineq_count)

            eq_count = eq_indices.size
            if eq_count > 0:
                inputs.append(f"nonlinear_equality_constraints = {eq_count}")
                inputs.append("nonlinear_equality_targets" + " 0.0" * eq_count)
                inputs.append("nonlinear_equality_scale_types 'value'")
                inputs.append("nonlinear_equality_scales" + " 1.0" * eq_count)
        return inputs

    def _start(self, initial_values: NDArray[np.float64]) -> None:
        pwd = Path.cwd()
        output_dir = create_output_path("dakota", self._output_dir)
        output_dir.mkdir()
        chdir(output_dir)

        try:
            self._start_direct_interface(initial_values)
        finally:
            if pwd.exists():
                chdir(pwd)

    def _start_direct_interface(self, initial_values: NDArray[np.float64]) -> None:
        driver = _DakotaDriver(
            self._config.optimizer,
            self._optimizer_callback,
            self._constraint_indices,
            self._get_inputs(initial_values),
        )
        try:
            driver.run_dakota(
                infile="dakota_Input.in",
                stdout="Report.txt",
                stderr="dakota_errors.txt",
            )
        except Exception as err:
            if driver.exception:
                raise driver.exception from err
            raise


# mypy: disallow-subclassing-any=False
class _DakotaDriver(DakotaBase):
    def __init__(
        self,
        optimizer_config: OptimizerConfig,
        optimizer_callback: OptimizerCallback,
        constraint_indices: _ConstraintIndices | None,
        inputs: dict[str, list[str]],
    ) -> None:
        self._optimizer_config = optimizer_config
        self._optimizer_callback = optimizer_callback
        self._split_evaluations = optimizer_config.split_evaluations
        self._constraint_indices = constraint_indices
        self.exception: Exception | None = None
        super().__init__(DakotaInput(**inputs))

    def dakota_callback(
        self,
        **kwargs: Any,  # noqa: ANN401
    ) -> dict[str, NDArray[np.float64]]:
        try:
            asv = [response for response in kwargs["asv"] if response > 0]
            if len(set(asv)) > 1:
                msg = "Non-unique ASV elements different from 0 are not yet supported."
                raise NotImplementedError(msg)  # noqa: TRY301
            return_functions = asv[0] in (1, 3)
            compute_gradients = asv[0] in (2, 3)
            function_result, gradient_result = _compute_response(
                np.concatenate(
                    (kwargs["cv"], kwargs["div"].astype(np.float64), kwargs["drv"]),
                ),
                self._constraint_indices,
                self._optimizer_callback,
                return_functions=return_functions,
                compute_gradients=compute_gradients,
                split_evaluations=self._split_evaluations,
            )
        except Exception as err:
            self.exception = err
            raise
        # Store the return value.
        retval = {}
        if kwargs["asv"][0] & 1:
            retval["fns"] = function_result
        if kwargs["asv"][0] & 2:
            retval["fnGrads"] = gradient_result
        return retval

    def run_dakota(
        self,
        infile: str = "dakota.in",
        stdout: str | None = None,
        stderr: str | None = None,
        restart: int = 0,
        throw_on_error: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        overridden_infile = self._override_input_file()
        if overridden_infile is None:
            self.input.write_input(infile, driver_instance=self)
        else:
            lines = []
            with Path(overridden_infile).open("r", encoding="utf-8") as fp_in:
                for line in fp_in:
                    idx = line.find("analysis_components")
                    if idx >= 0:
                        ident = str(id(self))
                        _USER_DATA[ident] = self
                        lines.append(line[:idx] + f"analysis_components = '{ident}'\n")
                    else:
                        lines.append(line)
            with Path(infile).open("w", encoding="utf-8") as fp_out:
                fp_out.writelines(lines)
        run_dakota(infile, stdout, stderr, restart, throw_on_error)

    def _override_input_file(self) -> str | None:
        input_file = None
        if isinstance(self._optimizer_config.options, list):
            input_file = next(
                (
                    option
                    for option in self._optimizer_config.options
                    if option.strip().startswith("input_file")
                ),
                None,
            )
        if input_file is not None:
            split_input_file = input_file.split("=", 1)
            if len(split_input_file) > 1:
                path = Path(split_input_file[1].strip())
                if path.is_file():
                    return str(path)
            msg = f"Invalid input_file option: {input_file}"
            raise RuntimeError(msg)
        return None


def _compute_response(  # noqa: PLR0913
    variables: NDArray[np.float64],
    constraint_indices: _ConstraintIndices | None,
    optimizer_callback: OptimizerCallback,
    *,
    return_functions: bool,
    compute_gradients: bool,
    split_evaluations: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if return_functions and compute_gradients and split_evaluations:
        functions, _ = optimizer_callback(
            variables, return_functions=True, return_gradients=False
        )
        _, gradients = optimizer_callback(
            variables, return_functions=False, return_gradients=True
        )
    else:
        functions, gradients = optimizer_callback(
            variables,
            return_functions=return_functions,
            return_gradients=compute_gradients,
        )

    # Reorder functions and gradients that correspond to nonlinear constraints:
    if constraint_indices is not None:
        ge_indices, le_indices, eq_indices = constraint_indices
        indices = np.hstack((0, le_indices + 1, ge_indices + 1, eq_indices + 1))
        if return_functions:
            functions = functions[indices]
        if compute_gradients:
            gradients = gradients[indices, :]
    return functions, gradients


class DakotaOptimizerPlugin(OptimizerPlugin):
    """Plugin class for optimization via Dakota."""

    def create(
        self, config: EnOptConfig, optimizer_callback: OptimizerCallback
    ) -> DakotaOptimizer:
        """Initialize the optimizer plugin.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """
        return DakotaOptimizer(config, optimizer_callback)

    def is_supported(self, method: str, *, explicit: bool) -> bool:  # noqa: ARG002
        """Check if a method is supported.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """
        return method.lower() in (_SUPPORTED_METHODS | {"default"})
