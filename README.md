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
