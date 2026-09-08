# fdc-unit-converter

A flexible unit conversion library for scientific and engineering applications.

## Installation

```bash
pip install fdc-unit-converter
```

## Usage

```python
from fdc_unit_converter import UnitConverter, units

# Convert 1000 meters to kilometers
result = UnitConverter.convert(1000, units.meter, units.kilometer)
print(result)  # 1.0
```

## Features
- Conversion across magnitudes: length, pressure, temperature, volume, etc.
- Works with scalars, lists, numpy arrays, and pandas Series.
- Every ordered pair of units within a magnitude converts, in both directions
  and independently of the path taken.

## Adding a unit

Each unit is defined by a factor relative to the base unit of its magnitude, so
declaring one line gives you its conversion to and from every other unit of that
magnitude:

```python
from fdc_unit_converter import Magnitude, Unit

yard = Unit("yard", "yd", Magnitude.LENGTH, 0.9144)   # 0.9144 m per yard
```

Temperature also needs an offset, expressed in the unit's own scale and applied
before the factor, so the declaration reads like the familiar formula:

```python
# (F - 32) * 5/9 = degrees celsius
fahrenheit = Unit("fahrenheit", "°F", Magnitude.TEMPERATURE, 5 / 9, -32)
```

A few relations are not affine. API gravity is reciprocal in specific gravity,
so it supplies its own pair of conversion functions instead of a factor:

```python
API_gravity = Unit(
    "API gravity", "°API", Magnitude.LIQUID_GRAVITY,
    to_base=lambda api: 141.5 / (api + 131.5),
    from_base=lambda sg: (141.5 / sg) - 131.5,
)
```

## A note on concentration

`ppm` is a mass fraction and `mg/L` is a mass concentration, so converting
between them depends on the density of the solution — for brines that is a
5–20% difference. They are therefore declared as two separate magnitudes
(`MASS_FRACTION` and `MASS_CONCENTRATION`) and the library will refuse to
convert between them rather than apply a fixed factor. Convert through density
explicitly in caller code.

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

```bash
# Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh      # Linux / macOS
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows

# Create the virtualenv and install all dependencies (including dev)
uv sync

# Run the test suite
uv run pytest

# Install the pre-commit hooks
uv run pre-commit install

# Build the distribution artifacts into dist/
uv build
```

Adding dependencies:

```bash
uv add <package>              # runtime dependency
uv add --dev <package>        # development dependency
```

## Releasing

`scripts/release.py` automates the whole publication flow: version bump, tests,
pre-commit, build, a clean-environment smoke test of the wheel, upload to PyPI,
and the git commit + tag + push.

```bash
export UV_PUBLISH_TOKEN=pypi-...      # PyPI API token

uv run scripts/release.py patch --dry-run   # rehearse without publishing
uv run scripts/release.py patch --test-pypi # publish to TestPyPI
uv run scripts/release.py minor             # publish to PyPI and tag vX.Y.Z
```

Accepts `major`, `minor`, `patch` or an explicit version (`1.2.3`). It refuses
to run on a dirty working tree or when the target tag already exists, and it
rolls `pyproject.toml` back if any step fails. Run `--help` for all flags.
