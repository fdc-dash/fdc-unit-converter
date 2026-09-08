"""
Whole-matrix properties of the conversion system, from ID-3546.

The pairwise tables the library used before left 77 of 202 ordered pairs
unconvertible, 17 of them asymmetric. These tests pin the properties that the
base-unit model gives us so the holes cannot come back:

* every ordered pair within a magnitude converts;
* every conversion is reversible;
* every conversion is independent of the path taken;
* the values released in 0.2.0 are preserved, except for four factors that were
  mutually inconsistent and had to be reconciled.
"""

import collections
import itertools
import json
import pathlib

import pytest

from fdc_unit_converter import Magnitude, Unit, UnitConverter
from fdc_unit_converter import units as u

BASELINE = json.loads((pathlib.Path(__file__).parent / "data" / "baseline_0_2_0.json").read_text())

# Factors that were internally inconsistent in 0.2.0 and now resolve to the
# value implied by the rest of their magnitude. See the acceptance criteria in
# ID-3546. Keys are (from, to); values are the relative change we accept.
RECONCILED = {
    ("pascal", "kilogram_per_square_centimeter"): 3e-4,
    ("pound_per_cubic_foot", "gram_per_cubic_centimeter"): 1e-4,
    ("stock_tank_barrel_per_day", "thousand_cubic_meter_per_day"): 1e-6,
}

# Everything else has to match 0.2.0 this closely.
BASELINE_TOLERANCE = 1e-4


def _units_by_magnitude():
    by_magnitude = collections.defaultdict(list)
    for name in dir(u):
        if name.startswith("_"):
            continue
        obj = getattr(u, name)
        if isinstance(obj, Unit):
            by_magnitude[obj.magnitude].append((name, obj))
    return by_magnitude


UNITS_BY_MAGNITUDE = _units_by_magnitude()
ALL_UNITS = [(name, unit) for units in UNITS_BY_MAGNITUDE.values() for name, unit in units]
ORDERED_PAIRS = [
    (magnitude, a_name, a, b_name, b)
    for magnitude, units in UNITS_BY_MAGNITUDE.items()
    for (a_name, a), (b_name, b) in itertools.permutations(units, 2)
]


def _ids(pairs):
    return [f"{magnitude}:{a}->{b}" for magnitude, a, _, b, _ in pairs]


# ------------------------------
# Completeness
# ------------------------------
@pytest.mark.parametrize("magnitude,a_name,a,b_name,b", ORDERED_PAIRS, ids=_ids(ORDERED_PAIRS))
def test_every_ordered_pair_converts(magnitude, a_name, a, b_name, b):
    result = UnitConverter.convert(1.0, a, b)
    assert result == result  # not NaN


def test_every_declared_magnitude_has_at_least_one_unit():
    declared = {value for name, value in vars(Magnitude).items() if not name.startswith("_") and isinstance(value, str)}
    assert declared - set(UNITS_BY_MAGNITUDE) == set()


# ------------------------------
# Reversibility
# ------------------------------
@pytest.mark.parametrize("magnitude,a_name,a,b_name,b", ORDERED_PAIRS, ids=_ids(ORDERED_PAIRS))
def test_conversion_is_reversible(magnitude, a_name, a, b_name, b):
    for value in (1.0, 7.5, 273.4):
        assert UnitConverter.convert(UnitConverter.convert(value, a, b), b, a) == pytest.approx(value, rel=1e-12)


# ------------------------------
# Path independence
# ------------------------------
@pytest.mark.parametrize("magnitude", sorted(UNITS_BY_MAGNITUDE), ids=sorted(UNITS_BY_MAGNITUDE))
def test_conversion_is_path_independent(magnitude):
    units = UNITS_BY_MAGNITUDE[magnitude]
    for (a_name, a), (b_name, b) in itertools.permutations(units, 2):
        direct = UnitConverter.convert(7.5, a, b)
        for c_name, c in units:
            if c is a or c is b:
                continue
            via_c = UnitConverter.convert(UnitConverter.convert(7.5, a, c), c, b)
            assert via_c == pytest.approx(direct, rel=1e-12), f"{a_name}->{b_name} differs via {c_name}"


# ------------------------------
# Non-regression against the released 0.2.0
# ------------------------------
@pytest.mark.parametrize(
    "magnitude,a_name,b_name,value,expected",
    BASELINE,
    ids=[f"{m}:{a}->{b}@{v}" for m, a, b, v, _ in BASELINE],
)
def test_matches_released_0_2_0(magnitude, a_name, b_name, value, expected):
    result = UnitConverter.convert(value, getattr(u, a_name), getattr(u, b_name))
    tolerance = RECONCILED.get((a_name, b_name), BASELINE_TOLERANCE)
    assert result == pytest.approx(expected, rel=tolerance)


@pytest.mark.parametrize("pair,tolerance", sorted(RECONCILED.items()))
def test_reconciled_factors_actually_moved(pair, tolerance):
    """The four documented factors changed, and by less than the budget we quoted."""
    a_name, b_name = pair
    expected = next(row[4] for row in BASELINE if row[1] == a_name and row[2] == b_name and row[3] == 1.0)
    result = UnitConverter.convert(1.0, getattr(u, a_name), getattr(u, b_name))
    assert result != expected
    assert result == pytest.approx(expected, rel=tolerance)


# ------------------------------
# The three conversions ID-3546 was blocked on
# ------------------------------
def test_priority_density_gcm3_to_kgm3():
    assert UnitConverter.convert(1.05, u.gram_per_cubic_centimeter, u.kilogram_per_cubic_meter) == pytest.approx(1050.0)


def test_priority_length_millimeter_to_meter():
    assert UnitConverter.convert(2500, u.millimeter, u.meter) == pytest.approx(2.5)
    assert UnitConverter.convert(2.5, u.meter, u.millimeter) == pytest.approx(2500.0)


def test_priority_viscosity_centipoise_to_poise():
    assert UnitConverter.convert(100, u.centipoise, u.poise) == pytest.approx(1.0)
    assert UnitConverter.convert(1, u.poise, u.centipoise) == pytest.approx(100.0)


def test_length_foot_to_inch_crosses_the_old_island_boundary():
    assert UnitConverter.convert(1, u.foot, u.inch) == pytest.approx(12.0)
    assert UnitConverter.convert(12, u.inch, u.foot) == pytest.approx(1.0)


# ------------------------------
# Concentration: the two families must stay apart
# ------------------------------
def test_mass_fraction_and_mass_concentration_are_different_magnitudes():
    """ppm is mass/mass and mg/L is mass/volume; there is no density-free factor."""
    assert u.part_per_million.magnitude != u.milligram_per_liter.magnitude
    with pytest.raises(ValueError, match="different magnitudes"):
        UnitConverter.convert(1000, u.part_per_million, u.milligram_per_liter)


def test_mass_concentration_within_family():
    assert UnitConverter.convert(1, u.gram_per_liter, u.milligram_per_liter) == pytest.approx(1000.0)
    assert UnitConverter.convert(1, u.picogram_per_milliliter, u.milligram_per_liter) == pytest.approx(1e-6)


def test_mass_fraction_within_family():
    assert UnitConverter.convert(1, u.percent_by_mass, u.part_per_million) == pytest.approx(10_000.0)
    assert UnitConverter.convert(1000, u.part_per_million, u.part_per_billion) == pytest.approx(1e6)


# ------------------------------
# New magnitudes requested in ID-3546
# ------------------------------
def test_new_units_cover_the_requested_measurements():
    assert UnitConverter.convert(3600, u.revolution_per_minute, u.hertz) == pytest.approx(60.0)
    assert UnitConverter.convert(1.5, u.kiloampere, u.ampere) == pytest.approx(1500.0)
    assert UnitConverter.convert(1, u.kilogram_force, u.newton) == pytest.approx(9.80665)
    assert UnitConverter.convert(1, u.gram_per_mole, u.kilogram_per_kilomole) == pytest.approx(1.0)
    assert UnitConverter.convert(1, u.cubic_meter, u.liter) == pytest.approx(1000.0)
    # A lean natural gas, ~1000 Btu/scf, is ~8900 kcal/m3.
    assert UnitConverter.convert(
        1000, u.british_thermal_unit_per_standard_cubic_feet, u.kilocalorie_per_cubic_meter
    ) == pytest.approx(8899, rel=1e-3)


# ------------------------------
# The escape hatch for non-affine relations
# ------------------------------
def test_api_gravity_is_not_affine_but_still_round_trips():
    for api in (10.0, 30.0, 45.0):
        sg = UnitConverter.convert(api, u.API_gravity, u.specific_gravity)
        assert sg == pytest.approx(141.5 / (api + 131.5))
        assert UnitConverter.convert(sg, u.specific_gravity, u.API_gravity) == pytest.approx(api)


def test_unit_rejects_a_half_declared_escape_hatch():
    with pytest.raises(ValueError, match="must define both"):
        Unit("broken", "x", Magnitude.LENGTH, to_base=lambda v: v)


# ------------------------------
# The magnitude registry introduced in 1.0.0
# ------------------------------
def test_list_magnitudes_covers_every_magnitude_with_units():
    assert set(Magnitude.list_magnitudes()) == set(UNITS_BY_MAGNITUDE)


def test_list_magnitude_units_matches_the_declared_units():
    listed = Magnitude.list_magnitude_units(Magnitude.LENGTH)
    assert {unit.symbol for unit in listed} == {unit.symbol for _, unit in UNITS_BY_MAGNITUDE[Magnitude.LENGTH]}


def test_list_magnitude_units_of_an_unknown_magnitude_is_empty():
    assert Magnitude.list_magnitude_units("not_a_magnitude") == []


# ------------------------------
# Adding a unit is one line
# ------------------------------
def test_a_newly_declared_unit_converts_against_every_existing_one():
    yard = Unit("yard", "yd", Magnitude.LENGTH, 0.9144)
    for name, existing in UNITS_BY_MAGNITUDE[Magnitude.LENGTH]:
        assert UnitConverter.convert(UnitConverter.convert(3.0, yard, existing), existing, yard) == pytest.approx(3.0)
    assert UnitConverter.convert(1, yard, u.foot) == pytest.approx(3.0)
    assert UnitConverter.convert(1, yard, u.inch) == pytest.approx(36.0)
