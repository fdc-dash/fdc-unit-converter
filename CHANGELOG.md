# Changelog

## 1.1.0

Resolves ID-3546. In 0.2.0, 77 of the 202 ordered unit pairs could not be
converted even though both units existed and declared the same magnitude, and
17 pairs converted in one direction only.

### Fixed

- **Every ordered pair within a magnitude now converts**, in both directions and
  independently of the path taken. Units are defined by a factor (and, for
  temperature, an offset) relative to the base unit of their magnitude, and
  every pair is derived from those, replacing the 14 hand-maintained tables of
  directed pairs.
- **`viscosity` now works.** `UnitConverter.convert` had no branch for it, so
  `centipoise ↔ poise` raised `Unknown magnitude 'viscosity'`.
- **`from src.fdc_unit_converter.magnitudes import Magnitude`**, which made
  1.0.0 unimportable from any install outside the source tree, is gone.
  `Magnitude` now lives in `fdc_unit_converter.magnitudes` and is re-exported
  from `fdc_unit_converter` and `fdc_unit_converter.units`, so all three import
  paths work.
- `GAS_GRAVITY` and `MASS` were declared in `Magnitude` with no units. Both now
  have units.

### Changed — conversion factors

Four factors were mutually inconsistent with the rest of their own magnitude:
`pascal → kg/cm² → pascal` returned `1.0002783` rather than `1`. Deriving every
pair from one factor per unit requires a single value, so these now resolve to
the value implied by the rest of their magnitude:

| Conversion | 0.2.0 | 1.1.0 | Relative change |
|---|---|---|---|
| `pascal → kilogram_per_square_centimeter` | `0.0000102` | `1.0197162129779e-05` | 2.8e-4 |
| `pound_per_cubic_foot → gram_per_cubic_centimeter` | `0.01602` | `0.0160184634` | 9.6e-5 |
| `stock_tank_barrel_per_day → thousand_cubic_meter_per_day` | `0.000158987295` | `1.589872132944e-04` | 5.1e-7 |
| `bar → pound_per_square_inch` and other `pressure` pairs | `14.5037738` | `14.503773765761` | ~2e-9 |

Every other conversion released in 0.2.0 is preserved to within `rel=1e-4`, and
is pinned by `tests/test_conversion_matrix.py` against a table captured from the
published 0.2.0 wheel.

`foot` and `inch` are now the exact international definitions (0.3048 m and
0.0254 m); the previous `1 / 3.280839895` differed by 4e-12.

### Added

New magnitudes and units for formation-water and gas laboratory data:

| Magnitude | Units |
|---|---|
| `MASS_CONCENTRATION` | `milligram_per_liter`, `microgram_per_liter`, `gram_per_liter`, `picogram_per_milliliter` |
| `MASS_FRACTION` | `part_per_million`, `part_per_billion`, `percent_by_mass` |
| `FREQUENCY` | `hertz`, `kilohertz`, `revolution_per_minute` |
| `ELECTRIC_CURRENT` | `ampere`, `milliampere`, `kiloampere` |
| `FORCE` | `newton`, `kilonewton`, `kilogram_force`, `pound_force` |
| `CALORIFIC_VALUE` | `kilocalorie_per_cubic_meter`, `megajoule_per_cubic_meter`, `british_thermal_unit_per_standard_cubic_feet` |
| `MOLAR_MASS` | `kilogram_per_kilomole`, `gram_per_mole` |
| `MASS` | `kilogram`, `gram`, `tonne`, `pound` |
| `GAS_GRAVITY` | `gas_specific_gravity` |

Added to existing magnitudes: `liter` and `liter_per_day`, `centimeter`,
`pascal_second` and `millipascal_second`.

**Concentration is deliberately split in two.** `ppm` is mass/mass and `mg/L` is
mass/volume, so converting between them depends on the density of the solution —
5–20% for brines. They are separate magnitudes, and the library raises
`Cannot convert units of different magnitudes` rather than applying a fixed
factor. Convert through density explicitly in caller code.

### API

- `Unit` takes three new optional arguments: `factor`, `offset`, and a
  `to_base`/`from_base` pair for relations that are not affine. The existing
  three-argument constructor is unchanged.
- The private `UnitConverter._convert_<magnitude>` methods are gone; they held
  the pairwise tables. `UnitConverter.convert` is unchanged.

## 1.0.0

Not importable from PyPI — see the fix above. Use 1.1.0.

## 0.2.0

Added `kelvin` and its temperature conversions.

## 0.1.0

Initial release.
