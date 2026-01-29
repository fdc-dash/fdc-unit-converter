# fdc-unit-converter

A flexible unit conversion library for scientific and engineering applications.

## Installation

```bash
pip install fdc-unit-converter
```

## Features
- Conversion across magnitudes: length, pressure, temperature, volume, etc.
- Works with scalars, lists, numpy arrays, and pandas Series.


## Usage

- Basic conversion example
```python
from fdc_unit_converter import UnitConverter, units

# Convert 1000 meters to kilometers
result = UnitConverter.convert(1000, units.meter, units.kilometer)
print(result)  # 1.0
```

<br>
- List available magnitudes

```python
from fdc_unit_converter.magnitudes import Magnitude
magnitudes = Magnitude.list_magnitudes()
print(magnitudes)
```

<br>
- List available units for a magnitude

```python
from fdc_unit_converter.magnitudes import Magnitude
units = Magnitude.list_magnitude_units(Magnitude.LENGTH)
print([(unit.name, unit.symbol)  for unit in units])
```
