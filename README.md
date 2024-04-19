# A Dakota optimizer plugin for `ropt`
This package installs a plugin for the `ropt` robust optimizer package, giving
access to algorithms from the Dakota optimization package.


## Dependencies
This code has been tested with Python versions 3.8, 3.9, 3.10 and 3.11.

The backend is based on the [Dakota](https://dakota.sandia.gov/) optimizer and
depends on the [Carolina](https://github.com/equinor/Carolina) Python wrapper.


## Installation
To install, enter the distribution directory and execute:

```bash
pip install .
```

## Running the tests
To run the test suite, install the necessary dependencies and execute `pytest`:

```bash
pip install .[test]
pytest
```
