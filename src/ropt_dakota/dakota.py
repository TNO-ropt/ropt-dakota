"""This module implements the Dakota optimization plugin."""

import re
from math import isfinite
from os import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Final, Literal

import numpy as np
from dakota import _USER_DATA, DakotaBase, DakotaInput, run_dakota
from numpy.typing import NDArray
from ropt.config import EnOptConfig
from ropt.config.options import OptionsSchemaModel
from ropt.exceptions import ConfigError
from ropt.optimization import OptimizerCallback
from ropt.plugins.optimizer.base import Optimizer, OptimizerPlugin
from ropt.plugins.optimizer.utils import (
    NormalizedConstraints,
    create_output_path,
    get_masked_linear_constraints,
)

_PRECISION: Final[int] = 8
_INF: Final = 1e30


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
    """Dakota optimization backend for ropt.

    This class provides an interface to several optimization algorithms from
    [`Dakota`](https://snl-dakota.github.io/), enabling their use within `ropt`.

    To select an optimizer, set the `method` field within the
    [`optimizer`][ropt.config.OptimizerConfig] section of the
    [`EnOptConfig`][ropt.config.EnOptConfig] configuration object to the
    desired algorithm's name. Most methods support the general options defined
    in the [`EnOptConfig`][ropt.config.EnOptConfig] object. For
    algorithm-specific options, use the `options` dictionary within the
    [`optimizer`][ropt.config.OptimizerConfig] section.

    The table below lists the included methods together with the method-specific
    options that are supported. Click on the method name to consult the
    corresponding [`Dakota`](https://snl-dakota.github.io/) documentation:

    --8<-- "dakota.md"
    """

    def __init__(
        self, config: EnOptConfig, optimizer_callback: OptimizerCallback
    ) -> None:
        """Initialize the optimizer implemented by the Dakota plugin.

        See the [ropt.plugins.optimizer.base.Optimizer][] abstract base class.

        # noqa
        """
        self._config = config
        self._optimizer_callback = optimizer_callback
        self._normalized_constraints: NormalizedConstraints | None
        self._output_dir: Path

        if self._config.nonlinear_constraints is not None:
            self._normalized_constraints = NormalizedConstraints()
            lower, upper = (
                self._config.nonlinear_constraints.lower_bounds,
                self._config.nonlinear_constraints.upper_bounds,
            )
            self._normalized_constraints.set_bounds(lower, upper)
        else:
            self._normalized_constraints = None

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
                + self._get_linear_constraints_section(initial_values)
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
        if self._config.gradient.evaluation_policy == "speculative":
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
        names = tuple(f"variable{idx}" for idx in range(initial_values.size))
        lower_bounds = self._config.variables.lower_bounds[self._config.variables.mask]
        upper_bounds = self._config.variables.upper_bounds[self._config.variables.mask]
        initial_values = initial_values[self._config.variables.mask]
        names = tuple(
            name
            for name, enabled in zip(names, self._config.variables.mask, strict=True)
            if enabled
        )
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
                f"{bound:{_PRECISION}f}" if isfinite(bound) else f"-{_INF}"
                for bound in lower_bounds
            ),
        )
        inputs.append(
            "upper_bounds "
            + " ".join(
                f"{bound:{_PRECISION}f}" if isfinite(bound) else f"+{_INF}"
                for bound in upper_bounds
            ),
        )
        inputs.append(
            "descriptors " + "  ".join(f"'{name}'" for name in names),
        )
        return inputs

    def _get_linear_constraints_section(
        self, initial_values: NDArray[np.float64]
    ) -> list[str]:
        inputs: list[str] = []

        if self._config.linear_constraints is not None:
            all_coefficients, all_lower_bounds, all_upper_bounds = (
                get_masked_linear_constraints(self._config, initial_values)
            )

            eq_idx = np.abs(all_lower_bounds - all_upper_bounds) <= 1e-15  # noqa: PLR2004
            ineq_idx = np.logical_not(eq_idx)

            if np.any(ineq_idx):
                coefficients = all_coefficients[ineq_idx, :]
                lower_bounds = all_lower_bounds[ineq_idx]
                upper_bounds = all_upper_bounds[ineq_idx]
                lower_bounds[all_lower_bounds < -_INF] = -_INF
                upper_bounds[all_upper_bounds > _INF] = _INF
                # Add 0.0 to prevent -0.0 values:
                coefficients += 0.0
                lower_bounds += 0.0
                upper_bounds += 0.0
                inputs.append(
                    "linear_inequality_constraint_matrix = "
                    + "\n".join(
                        " ".join(
                            f"{value:{_PRECISION}f}" for value in coefficients[idx, :]
                        )
                        for idx in range(lower_bounds.size)
                    ),
                )
                inputs.append(
                    "linear_inequality_lower_bounds = "
                    + " ".join(f"{value:{_PRECISION}f}" for value in lower_bounds),
                )
                inputs.append(
                    "linear_inequality_upper_bounds = "
                    + " ".join(f"{value:{_PRECISION}f}" for value in upper_bounds),
                )

            if np.any(eq_idx):
                coefficients = all_coefficients[eq_idx, :]
                bounds = all_lower_bounds[eq_idx]
                # Add 0.0 to prevent -0.0 values:
                coefficients += 0.0
                bounds += 0.0
                inputs.append(
                    "linear_equality_constraint_matrix = "
                    + "\n".join(
                        " ".join(
                            f"{value:{_PRECISION}f}" for value in coefficients[idx, :]
                        )
                        for idx in range(bounds.size)
                    ),
                )
                inputs.append(
                    "linear_equality_targets ="
                    + " ".join(f"{value:{_PRECISION}f}" for value in bounds),
                )

        return inputs

    def _get_responses_section(self) -> list[str]:
        inputs: list[str] = []
        inputs.append("objective_function_scale_types 'value'")
        inputs.append("objective_function_scales = 1.0")
        if self._normalized_constraints is not None:
            ineq = self._normalized_constraints.is_eq.count(False)
            if ineq > 0:
                inputs.append(f"nonlinear_inequality_constraints = {ineq} ")
                inputs.append(
                    "nonlinear_inequality_lower_bounds " + " ".join(["0.0"] * ineq)
                )
                inputs.append(
                    "nonlinear_inequality_upper_bounds " + " ".join([f"{_INF}"] * ineq)
                )
                inputs.append("nonlinear_inequality_scale_types 'value'")
                inputs.append("nonlinear_inequality_scales" + " 1.0" * ineq)
            eq = self._normalized_constraints.is_eq.count(True)
            if eq > 0:
                inputs.append(f"nonlinear_equality_constraints = {eq}")
                inputs.append("nonlinear_equality_targets " + " ".join(["0.0"] * eq))
                inputs.append("nonlinear_equality_scale_types 'value'")
                inputs.append("nonlinear_equality_scales" + " 1.0" * eq)
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
            self._config,
            self._optimizer_callback,
            self._normalized_constraints,
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
        config: EnOptConfig,
        optimizer_callback: OptimizerCallback,
        normalized_constraints: NormalizedConstraints | None,
        inputs: dict[str, list[str]],
    ) -> None:
        self._config = config
        self._optimizer_callback = optimizer_callback
        self._normalized_constraints = normalized_constraints
        self.exception: Exception | None = None
        super().__init__(DakotaInput(**inputs))

    def dakota_callback(
        self,
        **kwargs: Any,  # noqa: ANN401
    ) -> dict[str, NDArray[np.float64] | None]:
        try:
            asv = [response for response in kwargs["asv"] if response > 0]
            if len(set(asv)) > 1:
                msg = "Non-unique ASV elements different from 0 are not yet supported."
                raise NotImplementedError(msg)  # noqa: TRY301
            return_functions = asv[0] in (1, 3)
            compute_gradients = asv[0] in (2, 3)
            function_result, gradient_result = self._compute_response(
                np.concatenate(
                    (kwargs["cv"], kwargs["div"].astype(np.float64), kwargs["drv"]),
                ),
                return_functions=return_functions,
                compute_gradients=compute_gradients,
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
        if isinstance(self._config.optimizer.options, list):
            input_file = next(
                (
                    option
                    for option in self._config.optimizer.options
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

    def _compute_response(
        self,
        variables: NDArray[np.float64],
        *,
        return_functions: bool,
        compute_gradients: bool,
    ) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
        if (
            return_functions
            and compute_gradients
            and self._config.gradient.evaluation_policy == "separate"
        ):
            callback_result = self._optimizer_callback(
                variables, return_functions=True, return_gradients=False
            )
            functions = callback_result.functions
            callback_result = self._optimizer_callback(
                variables, return_functions=False, return_gradients=True
            )
            gradients = callback_result.gradients
        else:
            callback_result = self._optimizer_callback(
                variables,
                return_functions=return_functions,
                return_gradients=compute_gradients,
            )
            functions = callback_result.functions
            gradients = callback_result.gradients

        if (
            self._normalized_constraints is not None
            and callback_result.nonlinear_constraint_bounds is not None
        ):
            assert self._config.nonlinear_constraints is not None
            self._normalized_constraints.set_bounds(
                *callback_result.nonlinear_constraint_bounds
            )

            self._normalized_constraints.reset()
            neq = [
                idx
                for idx, eq in enumerate(self._normalized_constraints.is_eq)
                if not eq
            ]
            eq = [
                idx for idx, eq in enumerate(self._normalized_constraints.is_eq) if eq
            ]
            if return_functions:
                assert functions is not None
                self._normalized_constraints.set_constraints(functions[1:].transpose())
                assert self._normalized_constraints.constraints is not None
                functions = np.hstack(
                    (
                        functions[0],
                        self._normalized_constraints.constraints[neq + eq, 0],
                    )
                )
            if compute_gradients:
                assert gradients is not None
                self._normalized_constraints.set_gradients(gradients[1:, :])
                assert self._normalized_constraints.gradients is not None
                gradients = np.vstack(
                    (
                        gradients[:1, :],
                        self._normalized_constraints.gradients[neq + eq, :],
                    )
                )

        return functions, gradients


class DakotaOptimizerPlugin(OptimizerPlugin):
    """Plugin class for optimization via Dakota."""

    @classmethod
    def create(
        cls, config: EnOptConfig, optimizer_callback: OptimizerCallback
    ) -> DakotaOptimizer:
        """Initialize the optimizer plugin.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """
        return DakotaOptimizer(config, optimizer_callback)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Check if a method is supported.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """
        return method.lower() in (_SUPPORTED_METHODS | {"default"})

    @classmethod
    def validate_options(
        cls, method: str, options: dict[str, Any] | list[str] | None
    ) -> None:
        """Validate the options of a given method.

        See the [ropt.plugins.optimizer.base.OptimizerPlugin][] abstract base class.

        # noqa
        """
        if options is not None:
            if isinstance(options, dict):
                msg = "Dakota optimizer options must be a list of strings"
                raise TypeError(msg)
            options_dict: dict[str, Any] = {}
            for option in options:
                split_option = re.split(r"\s*=\s*|\s+", option.strip(), maxsplit=1)
                options_dict[split_option[0]] = (
                    split_option[1]
                    if len(split_option) > 1 and split_option[1].strip()
                    else True
                )
            OptionsSchemaModel.model_validate(_OPTIONS_SCHEMA).get_options_model(
                method
            ).model_validate(options_dict)


_OPTIONS_SCHEMA: dict[str, Any] = {
    "methods": {
        "optpp_q_newton": {
            "options": {
                "search_method": Literal[
                    "value_based_line_search",
                    "gradient_based_line_search",
                    "trust_region",
                    "tr_pds",
                ],
                "merit_function": Literal["el_bakry", "argaez_tapia", "van_shanno"],
                "steplength_to_boundary": float,
                "centering_parameter": float,
                "max_step": float,
                "gradient_tolerance": float,
                "max_iterations": int,
                "convergence_tolerance": float,
                "max_function_evaluations": int,
            },
            "url": "https://snl-dakota.github.io/docs/6.21.0/users/usingdakota/reference/method-optpp_q_newton.html",
        },
        "conmin_mfd": {
            "options": {
                "max_iterations": int,
                "convergence_tolerance": float,
                "constraint_tolerance": float,
                "max_function_evaluations": int,
            },
            "url": "https://snl-dakota.github.io/docs/6.21.0/users/usingdakota/reference/method-conmin_mfd.html",
        },
        "conmin_frcg": {
            "options": {
                "max_iterations": int,
                "convergence_tolerance": float,
                "constraint_tolerance": float,
                "max_function_evaluations": int,
            },
            "url": "https://snl-dakota.github.io/docs/6.21.0/users/usingdakota/reference/method-conmin_frcg.html",
        },
        "mesh_adaptive_search": {
            "options": {
                "initial_delta": float,
                "variable_tolerance": float,
                "function_precision": float,
                "seed": int,
                "history_file": str,
                "display_format": str,
                "variable_neighborhood_search": float,
                "neighbor_order": int,
                "display_all_evaluations": bool,
                "use_surrogate": Literal["inform_search", "optimize"],
                "max_iterations": int,
                "max_function_evaluations": int,
            },
            "url": "https://snl-dakota.github.io/docs/6.21.0/users/usingdakota/reference/method-mesh_adaptive_search.html",
        },
        "coliny_ea": {
            "options": {
                "population_size": int,
                "initialization_type": Literal["simple_random", "unique_random"],
                "fitness_type": Literal["linear_rank", "merit_function"],
                "replacement_type": Literal[
                    "random", "chc", "elitist", "new_solutions_generated"
                ],
                "crossover_rate": float,
                "crossover_type": Literal["two_point", "blend", "uniform"],
                "mutation_rate": float,
                "mutation_type": Literal[
                    "replace_uniform",
                    "offset_normal",
                    "offset_cauchy",
                    "offset_uniform",
                    "non_adaptive",
                ],
                "constraint_penalty": float,
                "solution_target": float,
                "seed": int,
                "show_misc_options": str,
                "misc_options": None,
                "max_iterations": str,
                "convergence_tolerance": float,
                "max_function_evaluations": int,
            },
            "url": "https://snl-dakota.github.io/docs/6.21.0/users/usingdakota/reference/method-coliny_ea.html",
        },
        "soga": {
            "options": {
                "fitness_type": Literal["merit_function", "constraint_penalty"],
                "replacement_type": Literal[
                    "elitist",
                    "favor_feasible",
                    "roulette_wheel",
                    "unique_roulette_wheel",
                ],
                "convergence_type": Literal[
                    "best_fitness_tracker", "average_fitness_tracker"
                ],
                "max_iterations": int,
                "max_function_evaluations": int,
                "population_size": int,
                "print_each_pop": bool,
                "initialization_type": Literal["simple_random", "unique_random"],
                "crossover_type": Literal[
                    "multi_point_binary",
                    "multi_point_parameterized_binary",
                    "multi_point_real",
                    "shuffle_random",
                ],
                "crossover_rate": float,
                "mutation_type": Literal[
                    "bit_random",
                    "replace_uniform",
                    "offset_normal",
                    "offset_cauchy",
                    "offset_uniform",
                ],
                "mutation_rate": float,
                "seed": int,
                "convergence_tolerance": float,
            },
            "url": "https://snl-dakota.github.io/docs/6.21.0/users/usingdakota/reference/method-soga.html",
        },
        "moga": {
            "options": {
                "fitness_type": Literal["layer_rank", "domination_count"],
                "replacement_type": Literal[
                    "elitist", "roulette_wheel", "unique_roulette_wheel", "below_limit"
                ],
                "niching_type": Literal["radial,distance", "max_designs"],
                "convergence_type": Literal[
                    "metric_tracker", "percent_change", "num_generations"
                ],
                "postprocessor_type": Literal["orthogonal_distance"],
                "max_iterations": int,
                "max_function_evaluations": int,
                "population_size": int,
                "print_each_pop": bool,
                "initialization_type": Literal[
                    "simple_random", "unique_random", "flat_file"
                ],
                "crossover_type": Literal[
                    "multi_point_binary",
                    "multi_point_parameterized_binary",
                    "multi_point_real",
                    "shuffle_random",
                ],
                "crossover_rate": float,
                "mutation_type": Literal[
                    "bit_random",
                    "replace_uniform",
                    "offset_normal",
                    "offset_cauchy",
                    "offset_uniform",
                ],
                "mutation_rate": float,
                "seed": int,
                "convergence_tolerance": float,
            },
            "url": "https://snl-dakota.github.io/docs/6.21.0/users/usingdakota/reference/method-moga.html",
        },
        "asynch_pattern_search": {
            "options": {
                "initial_delta": float,
                "contraction_factor": float,
                "variable_tolerance": float,
                "solution_target": float,
                "merit_function": Literal[
                    "merit_max",
                    "merit_max_smooth",
                    "merit1",
                    "merit1_smooth",
                    "merit2",
                    "merit2_smooth",
                    "merit2_squared",
                ],
                "constraint_penalty": float,
                "smoothing_factor": float,
                "constraint_tolerance": float,
                "max_function_evaluations": int,
            },
            "url": "https://snl-dakota.github.io/docs/6.21.0/users/usingdakota/reference/method-asynch_pattern_search.html",
        },
    },
}

if __name__ == "__main__":
    from ropt.config.options import gen_options_table

    with Path("dakota.md").open("w", encoding="utf-8") as fp:
        fp.write(gen_options_table(_OPTIONS_SCHEMA))
