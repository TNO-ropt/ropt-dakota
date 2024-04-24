# A Dakota optimizer plugin for `ropt`
This package installs a plugin for the `ropt` robust optimization package,
providing access to algorithms from the Dakota optimization package.

`ropt-dakota` is developed by the Netherlands Organisation for Applied
Scientific Research (TNO). All files in this repository are released under the
GNU General Public License v3.0 (a copy is provided in the LICENSE file).


## Dependencies
This code has been tested with Python versions 3.8-3.12.

The plugin is based on the [Dakota](https://dakota.sandia.gov/) optimizer and
depends on the [Carolina](https://github.com/equinor/Carolina) Python wrapper.


## Installation
From PyPI:
```bash
pip install ropt-dakota
```


## Development
The `ropt-dakota` source distribution can be found on
[GitHub](https://github.com/tno-ropt/ropt-dakota). To install from source, enter
the distribution directory and execute:

```bash
pip install .
```


## Running the tests
To run the test suite, install the necessary dependencies and execute `pytest`:

```bash
pip install .[test]
pytest
```
