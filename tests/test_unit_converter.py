import pytest
import numpy as np
import pandas as pd

from fdc_unit_converter import UnitConverter, units


# ------------------------------
# Basic length conversions
# ------------------------------
def test_length_meter_to_kilometer():
    assert UnitConverter.convert(1000, units.meter, units.kilometer) == pytest.approx(1.0)


def test_length_kilometer_to_foot():
    assert UnitConverter.convert(1, units.kilometer, units.foot) == pytest.approx(3280.839895)


def test_length_inch_to_mm():
    assert UnitConverter.convert(1, units.inch, units.millimeter) == pytest.approx(25.4)


# ------------------------------
# Temperature conversions
# ------------------------------
def test_temperature_c_to_f():
    assert UnitConverter.convert(0, units.celsius, units.fahrenheit) == 32


def test_temperature_f_to_c():
    assert UnitConverter.convert(32, units.fahrenheit, units.celsius) == 0


def test_temperature_rankine_to_c():
    result = UnitConverter.convert(491.67, units.rankine, units.celsius)
    assert result == pytest.approx(0.0)


# ------------------------------
# Pressure conversions
# ------------------------------
def test_pressure_bar_to_psi():
    result = UnitConverter.convert(1, units.bar, units.pound_per_square_inch)
    assert result == pytest.approx(14.5037738)


def test_pressure_kgcm2_to_bar():
    result = UnitConverter.convert(1, units.kilogram_per_square_centimeter, units.bar)
    assert result == pytest.approx(0.980665, rel=1e-6)


# ------------------------------
# Volume conversions
# ------------------------------
def test_volume_m3_to_bbl():
    result = UnitConverter.convert(1, units.cubic_meter, units.barrel)
    assert result == pytest.approx(6.289814)


def test_volume_bbl_to_m3():
    result = UnitConverter.convert(1, units.barrel, units.cubic_meter)
    assert result == pytest.approx(1 / 6.289814)


# ------------------------------
# Rate conversions
# ------------------------------
def test_rate_cmd_to_stbd():
    result = UnitConverter.convert(1, units.cubic_meter_per_day, units.stock_tank_barrel_per_day)
    assert result == pytest.approx(6.289814)


def test_rate_stbd_to_cmd():
    result = UnitConverter.convert(6.289814, units.stock_tank_barrel_per_day, units.cubic_meter_per_day)
    assert result == pytest.approx(1.0, rel=1e-6)


# ------------------------------
# Currency conversions
# ------------------------------
def test_currency_dollar_to_musd():
    result = UnitConverter.convert(1_000_000, units.dollar, units.million_dollar)
    assert result == pytest.approx(1.0)


def test_currency_musd_to_dollar():
    result = UnitConverter.convert(1, units.million_dollar, units.dollar)
    assert result == 1_000_000


# ------------------------------
# Time conversions
# ------------------------------
def test_time_hour_to_sec():
    assert UnitConverter.convert(1, units.hour, units.second) == 3600


def test_time_day_to_hour():
    assert UnitConverter.convert(1, units.day, units.hour) == 24


# ------------------------------
# Density conversions
# ------------------------------
def test_density_lbft3_to_kgm3():
    assert UnitConverter.convert(1, units.pound_per_cubic_foot, units.kilogram_per_cubic_meter) == pytest.approx(16.0184634)


# ------------------------------
# Permeability conversions
# ------------------------------
def test_perm_darcy_to_millidarcy():
    assert UnitConverter.convert(1, units.darcy, units.millidarcy) == 1000


# ------------------------------
# Liquid gravity conversions
# ------------------------------
def test_liquid_gravity_api_to_sg():
    result = UnitConverter.convert(35.0, units.API_gravity, units.specific_gravity)
    expected = 141.5 / (35.0 + 131.5)
    assert result == pytest.approx(expected)


def test_liquid_gravity_sg_to_api():
    sg = 0.85
    result = UnitConverter.convert(sg, units.specific_gravity, units.API_gravity)
    expected = (141.5 / sg) - 131.5
    assert result == pytest.approx(expected)


# ------------------------------
# Array and Series support
# ------------------------------
def test_list_conversion_to_numpy():
    values = [1, 2, 3]
    result = UnitConverter.convert(values, units.minute, units.second)
    assert isinstance(result, np.ndarray)
    assert all(result == np.array([60, 120, 180]))


def test_numpy_array_conversion():
    arr = np.array([1, 2])
    result = UnitConverter.convert(arr, units.hour, units.minute)
    assert all(result == np.array([60, 120]))


def test_pandas_series_conversion():
    series = pd.Series([1, 2])
    result = UnitConverter.convert(series, units.hour, units.minute)
    assert all(result == pd.Series([60, 120]))


# ------------------------------
# Error handling
# ------------------------------
def test_incompatible_magnitudes():
    with pytest.raises(ValueError, match="different magnitudes"):
        UnitConverter.convert(1, units.meter, units.kilogram_per_cubic_meter)


def test_invalid_type():
    with pytest.raises(ValueError, match="non-numeric"):
        UnitConverter.convert("abc", units.meter, units.kilometer)


def test_none_units():
    with pytest.raises(ValueError, match="None unit"):
        UnitConverter.convert(1, None, units.meter)


def test_unsupported_conversion():
    with pytest.raises(ValueError, match="Cannot convert"):
        UnitConverter.convert(1, units.meter, units.acre)
