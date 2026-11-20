# A Dakota plugin for `ropt`

The `ropt-dakota` package extends the [`ropt`](https://tno-ropt.github.io/ropt/dev/)
module by providing a plugin that integrates optimization algorithms from the
[Dakota](https://dakota.sandia.gov/) toolkit. `ropt` itself is a robust
optimization framework designed for both continuous and discrete optimization
workflows and is extensible through its plugin architecture. Installing
`ropt-dakota` makes these Dakota algorithms directly available within `ropt`.

## Usage

For `asynch_pattern_search`, leave `backend.max_iterations` unset or `None`;
any other value raises a validation error. To limit function evaluations
instead, add `"max_function_evaluations = 100"` to the `backend.options` list.

## Reference

::: ropt_dakota.dakota.DakotaBackend
    options:
        members: False
