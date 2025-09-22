from typing import List, Union

import numpy as np
import pandas as pd

from . import units
from .units import Magnitude, Unit


class UnitConverter:
    """
    Convert value from one unit to another.
    """

    @staticmethod
    def convert(
        value: Union[int, float, List[Union[int, float]], np.ndarray, pd.Series],
        _from: Unit,
        _to: Unit,
    ) -> Union[int, float, np.ndarray, pd.Series]:
        """
        Convert value from '_from' to '_to' unit.
        List will be converted to numpy array.
        """
        UnitConverter._check_convertibility(value, _from, _to)

        value = UnitConverter._list_to_np_array(value)

        if _from.magnitude == Magnitude.AREA:
            return UnitConverter._convert_area(value, _from, _to)

        if _from.magnitude == Magnitude.COMPRESSIBILITY:
            return UnitConverter._convert_compressibility(value, _from, _to)

        if _from.magnitude == Magnitude.CURRENCY:
            return UnitConverter._convert_currency(value, _from, _to)

        if _from.magnitude == Magnitude.DENSITY:
            return UnitConverter._convert_density(value, _from, _to)

        if _from.magnitude == Magnitude.ENERGY:
            return UnitConverter._convert_energy(value, _from, _to)

        if _from.magnitude == Magnitude.LENGTH:
            return UnitConverter._convert_length(value, _from, _to)

        if _from.magnitude == Magnitude.LIQUID_GRAVITY:
            return UnitConverter._convert_liquid_gravity(value, _from, _to)

        if _from.magnitude == Magnitude.PERMEABILITY:
            return UnitConverter._convert_permeability(value, _from, _to)

        if _from.magnitude == Magnitude.PRESSURE:
            return UnitConverter._convert_pressure(value, _from, _to)

        if _from.magnitude == Magnitude.RATE:
            return UnitConverter._convert_rate(value, _from, _to)

        if _from.magnitude == Magnitude.TEMPERATURE:
            return UnitConverter._convert_temperature(value, _from, _to)

        if _from.magnitude == Magnitude.TIME:
            return UnitConverter._convert_time(value, _from, _to)

        if _from.magnitude == Magnitude.VOLUME:
            return UnitConverter._convert_volume(value, _from, _to)

        if _from.magnitude == Magnitude.VOLUME_RATIO:
            return UnitConverter._convert_volume_ratio(value, _from, _to)

        raise ValueError(f"Unknown magnitude '{_from.magnitude}'")

    @staticmethod
    def _check_convertibility(
        value: Union[int, float, List[Union[int, float]], np.ndarray, pd.Series],
        _from: Unit,
        _to: Unit,
    ) -> bool:
        """
        Check if conversion from '_from' to '_to' is possible.
        """
        ADMITED_MEMBER_TYPES = (int, float, np.number)
        ADMITED_TYPES = ADMITED_MEMBER_TYPES + (List, np.ndarray, pd.Series)

        if not isinstance(value, ADMITED_TYPES):
            raise ValueError("Cannot convert non-numeric values")

        if isinstance(value, List):
            if not all(isinstance(v, ADMITED_MEMBER_TYPES) for v in value):
                raise ValueError("Cannot convert non-numeric values")

        if isinstance(value, np.ndarray):
            if not np.issubdtype(value.dtype, np.number):
                raise ValueError("Cannot convert non-numeric values")

        if isinstance(value, pd.Series):
            if not all(isinstance(v, ADMITED_MEMBER_TYPES) for v in value.values):
                raise ValueError("Cannot convert non-numeric values")

        if _from is None or _to is None:
            raise ValueError("Cannot convert None unit")

        if not (isinstance(_from, Unit) and isinstance(_to, Unit)):
            raise ValueError("Cannot convert non-unit objects")

        if _from.magnitude != _to.magnitude:
            raise ValueError("Cannot convert units of different magnitudes")

    @staticmethod
    def _list_to_np_array(
        value: Union[int, float, List[Union[int, float]], np.ndarray, pd.Series]
    ) -> Union[int, float, np.ndarray]:
        """
        Convert list to numpy array.
        This method is private and should not be called directly.
        """
        if isinstance(value, list):
            return np.array(value)
        return value

    @staticmethod
    def _raise_cannot_convert_error(_from: Unit, _to: Unit) -> None:
        """
        Raise ValueError with message that conversion from _from to _to is not supported.
        This method is private and should not be called directly.
        """
        raise ValueError(f"Cannot convert from '{_from.name}' to '{_to.name}'")

    @staticmethod
    def _convert_compressibility(
        value: Union[int, float, np.ndarray, pd.Series],
        _from: Unit,
        _to: Unit,
    ) -> Union[int, float, np.ndarray, pd.Series]:
        """
        Convert compressibility value from '_from' to '_to'.
        This method is private and should not be called directly.
        """
        if value is None or _from == _to:
            return value

        # fmt: off
        # Syntax: (_from, _to): factor
        factor = {
            # From inverse_pound_per_square_inch
            (units.inverse_pound_per_square_inch, units.inverse_kilogram_per_square_centimeter): 14.2233433,
        }
        # fmt: on

        try:
            return value * factor[(_from, _to)]
        except KeyError:
            UnitConverter._raise_cannot_convert_error(_from, _to)

    @staticmethod
    def _convert_currency(
        value: Union[int, float, np.ndarray, pd.Series],
        _from: Unit,
        _to: Unit,
    ) -> Union[int, float, np.ndarray, pd.Series]:
        """
        Convert currency value from '_from' to '_to'.
        This method is private and should not be called directly.
        """
        if value is None or _from == _to:
            return value

        # fmt: off
        # Syntax: (_from, _to): factor
        factor = {
            (units.dollar, units.thousand_dollar): 1 / 1000,
            (units.dollar, units.million_dollar): 1 / 1_000_000,
            (units.thousand_dollar, units.dollar): 1000,
            (units.thousand_dollar, units.million_dollar): 1 / 1_000,
            (units.million_dollar, units.dollar): 1_000_000,
            (units.million_dollar, units.thousand_dollar): 1000,
        }
        # fmt: on

        try:
            return value * factor[(_from, _to)]
        except KeyError:
            UnitConverter._raise_cannot_convert_error(_from, _to)

    @staticmethod
    def _convert_density(
        value: Union[int, float, np.ndarray, pd.Series],
        _from: Unit,
        _to: Unit,
    ) -> Union[int, float, np.ndarray, pd.Series]:
        """
        Convert density value from '_from' to '_to'.
        This method is private and should not be called directly.
        """
        if value is None or _from == _to:
            return value

        # fmt: off
        # Syntax: (_from, _to): factor
        factor = {
            # From pound_per_cubic_foot
            (units.pound_per_cubic_foot, units.kilogram_per_cubic_meter): 16.0184634,
            (units.pound_per_cubic_foot, units.gram_per_cubic_centimeter): 0.01602,
        }
        # fmt: on

        try:
            return value * factor[(_from, _to)]
        except KeyError:
            UnitConverter._raise_cannot_convert_error(_from, _to)

    @staticmethod
    def _convert_length(
        value: Union[int, float, np.ndarray, pd.Series],
        _from: Unit,
        _to: Unit,
    ) -> Union[int, float, np.ndarray, pd.Series]:
        """
        Convert length value from '_from' to '_to'.
        This method is private and should not be called directly.
        """
        if value is None or _from == _to:
            return value

        # fmt: off
        # Syntax: (_from, _to): factor
        factor = {
            # From meter
            (units.meter, units.kilometer): 1 / 1000,
            (units.meter, units.foot): 3.280839895,
            (units.kilometer, units.meter): 1000,
            (units.kilometer, units.foot): 3280.839895,

            # From foot
            (units.foot, units.meter): 1 / 3.280839895,
            (units.foot, units.kilometer): 1 / 3280.839895,

            # From inch
            (units.inch, units.millimeter): 25.4,

            # From mm
            (units.millimeter, units.inch): 1 / 25.4,
        }
        # fmt: on

        try:
            return value * factor[(_from, _to)]
        except KeyError:
            UnitConverter._raise_cannot_convert_error(_from, _to)

    @staticmethod
    def _convert_liquid_gravity(
        value: Union[int, float, np.ndarray, pd.Series],
        _from: Unit,
        _to: Unit,
    ) -> Union[int, float, np.ndarray, pd.Series]:
        """
        Convert liquid gravity value from '_from' to '_to'.
        This method is private and should not be called directly.
        """
        if value is None or _from == _to:
            return value

        # fmt: off
        def formula(value, from_to):
            # Syntax: (_from, _to)
            f = {
                # From API_gravity
                (units.API_gravity, units.specific_gravity): 141.5 / (value + 131.5),

                # From specific_gravity
                (units.specific_gravity, units.API_gravity): (141.5 / value) - 131.5,
            }
            return f[from_to]
        # fmt: on

        try:
            return formula(value, (_from, _to))
        except KeyError:
            UnitConverter._raise_cannot_convert_error(_from, _to)

    @staticmethod
    def _convert_permeability(
        value: Union[int, float, np.ndarray, pd.Series],
        _from: Unit,
        _to: Unit,
    ) -> Union[int, float, np.ndarray, pd.Series]:
        """
        Convert permeability value from '_from' to '_to'.
        This method is private and should not be called directly.
        """
        if value is None or _from == _to:
            return value

        # fmt: off
        # Syntax: (_from, _to): factor
        factor = {
            # From darcy
            (units.darcy, units.millidarcy): 1_000,
            (units.darcy, units.microdarcy): 1_000_000,
            (units.darcy, units.nanodarcy): 1_000_000_000,
            (units.darcy, units.square_meter_permeability): 9.869233e-13,

            # From millidarcy
            (units.millidarcy, units.darcy): 1 / 1_000,
            (units.millidarcy, units.microdarcy): 1_000,
            (units.millidarcy, units.nanodarcy): 1_000_000,
            (units.millidarcy, units.square_meter_permeability): 9.869233e-16,

            # From microdarcy
            (units.microdarcy, units.darcy): 1 / 1_000_000,
            (units.microdarcy, units.millidarcy): 1 / 1_000,
            (units.microdarcy, units.nanodarcy): 1_000,
            (units.microdarcy, units.square_meter_permeability): 9.869233e-19,

            # From nanodarcy
            (units.nanodarcy, units.darcy): 1 / 1_000_000_000,
            (units.nanodarcy, units.millidarcy): 1 / 1_000_000,
            (units.nanodarcy, units.microdarcy): 1 / 1_000,
            (units.nanodarcy, units.square_meter_permeability): 9.869233e-22,

            # From square_meter_permeability
            (units.square_meter_permeability, units.darcy): 1 / 9.869233e-13,
            (units.square_meter_permeability, units.millidarcy): 1 / 9.869233e-16,
            (units.square_meter_permeability, units.microdarcy): 1 / 9.869233e-19,
            (units.square_meter_permeability, units.nanodarcy): 1 / 9.869233e-22,
        }
        # fmt: on

        try:
            return value * factor[(_from, _to)]
        except KeyError:
            UnitConverter._raise_cannot_convert_error(_from, _to)

    @staticmethod
    def _convert_pressure(
        value: Union[int, float, np.ndarray, pd.Series],
        _from: Unit,
        _to: Unit,
    ) -> Union[int, float, np.ndarray, pd.Series]:
        """
        Convert pressure value from '_from' to '_to'.
        This method is private and should not be called directly.
        """
        if value is None or _from == _to:
            return value

        # fmt: off
        # Syntax: (_from, _to): factor
        factor = {
            # From bar
            (units.bar, units.pascal): 100_000,
            (units.bar, units.kilopascal): 100,
            (units.bar, units.pound_per_square_inch): 14.5037738,
            (units.bar, units.kilogram_per_square_centimeter): 1.0197162129779,

            # From kilogram_per_square_centimeter
            (units.kilogram_per_square_centimeter, units.bar): 0.980665,
            (units.kilogram_per_square_centimeter, units.pound_per_square_inch): 14.2233433,
            (units.kilogram_per_square_centimeter, units.pascal): 98066.5,
            (units.kilogram_per_square_centimeter, units.kilopascal): 98.0665,

            # From pascal
            (units.pascal, units.bar): 1 / 100_000,
            (units.pascal, units.pound_per_square_inch): 0.000145037738,
            (units.pascal, units.kilogram_per_square_centimeter): 0.0000102,
            (units.pascal, units.kilopascal): 1 / 1_000,
            (units.kilopascal, units.bar): 1 / 100,
            (units.kilopascal, units.pascal): 1_000,
            (units.kilopascal, units.pound_per_square_inch): 0.1450377377,
            (units.kilopascal, units.kilogram_per_square_centimeter): 0.0101971621,

            # From pound_per_square_inch
            (units.pound_per_square_inch, units.bar): 0.0689475729,
            (units.pound_per_square_inch, units.kilogram_per_square_centimeter): 1 / 14.2233433,
            (units.pound_per_square_inch, units.pascal): 6894.75729,
            (units.pound_per_square_inch, units.kilopascal): 6.89475729,
        }
        # fmt: on

        try:
            return value * factor[(_from, _to)]
        except KeyError:
            UnitConverter._raise_cannot_convert_error(_from, _to)

    @staticmethod
    def _convert_temperature(
        value: Union[int, float, np.ndarray, pd.Series],
        _from: Unit,
        _to: Unit,
    ) -> Union[int, float, np.ndarray, pd.Series]:
        """
        Convert temperature value from '_from' to '_to'.
        This method is private and should not be called directly.
        """
        if value is None or _from == _to:
            return value

        # fmt: off
        def formula(value, from_to):
            # Syntax: (_from, _to)
            f = {
                # From celsius
                (units.celsius, units.fahrenheit): value * 9 / 5 + 32,

                # From fahrenheit
                (units.fahrenheit, units.celsius): (value - 32) * 5 / 9,
                (units.fahrenheit, units.rankine): value + 459.67,

                # From rankine
                (units.rankine, units.celsius): (value - 491.67) * 5 / 9,
            }
            return f[from_to]
        # fmt: on

        try:
            return formula(value, (_from, _to))
        except KeyError:
            UnitConverter._raise_cannot_convert_error(_from, _to)

    @staticmethod
    def _convert_time(
        value: Union[int, float, np.ndarray, pd.Series],
        _from: Unit,
        _to: Unit,
    ) -> Union[int, float, np.ndarray, pd.Series]:
        """
        Convert time value from '_from' to '_to'.
        This method is private and should not be called directly.
        """
        if value is None or _from == _to:
            return value

        # fmt: off
        # Syntax: (_from, _to): factor
        factor = {
            # From hour
            (units.hour, units.day): 1 / 24,
            (units.hour, units.minute): 60,
            (units.hour, units.second): 3600,

            # From day
            (units.day, units.hour): 24,
            (units.day, units.minute): 24 * 60,
            (units.day, units.second): 24 * 3600,

            # From minute
            (units.minute, units.hour): 1 / 60,
            (units.minute, units.day): 1 / (24 * 60),
            (units.minute, units.second): 60,

            # From second
            (units.second, units.hour): 1 / 3600,
            (units.second, units.day): 1 / (24 * 3600),
            (units.second, units.minute): 1 / 60,
        }
        # fmt: on

        try:
            return value * factor[(_from, _to)]
        except KeyError:
            UnitConverter._raise_cannot_convert_error(_from, _to)

    @staticmethod
    def _convert_rate(
        value: Union[int, float, np.ndarray, pd.Series],
        _from: Unit,
        _to: Unit,
    ) -> Union[int, float, np.ndarray, pd.Series]:
        """
        Convert rate value from '_from' to '_to'.
        This method is private and should not be called directly.
        """
        if value is None or _from == _to:
            return value

        # fmt: off
        # Syntax: (_from, _to): factor
        factor = {
            # From cubic_meter_per_day
            (units.cubic_meter_per_day, units.stock_tank_barrel_per_day): 6.289814,
            (units.cubic_meter_per_day, units.thousand_cubic_meter_per_day): 1 / 1000,
            (units.cubic_meter_per_day, units.standard_cubic_feet_per_day): 35.3146667,
            (units.cubic_meter_per_day, units.cubic_meter_per_second): 1 / 86_400,
            (units.thousand_cubic_meter_per_day, units.cubic_meter_per_day): 1000,
            (units.thousand_cubic_meter_per_day, units.stock_tank_barrel_per_day): 6289.814,

            # From cubic_meter_per_second
            (units.cubic_meter_per_second, units.cubic_meter_per_day): 86_400,
            (units.cubic_meter_per_second, units.thousand_cubic_meter_per_day): 86_400 / 1000,
            (units.cubic_meter_per_second, units.stock_tank_barrel_per_day): 86_400 * 6.289814,

            # From stock_tank_barrel_per_day
            (units.stock_tank_barrel_per_day, units.cubic_meter_per_day): 1 / 6.289814,
            (units.stock_tank_barrel_per_day, units.thousand_cubic_meter_per_day): 0.000158987295,

            # From standard_cubic_feet_per_day
            (units.standard_cubic_feet_per_day, units.cubic_meter_per_day): 1 / 35.3146667,
            (units.standard_cubic_feet_per_day, units.cubic_meter_per_second): 1 / 35.3146667 / 86_400,
            (units.standard_cubic_feet_per_day, units.thousand_cubic_meter_per_day): 1 / 35.3146667 / 1000,
            (units.standard_cubic_feet_per_day, units.thousand_standard_cubic_feet_per_day): 1 / 1000,
            (units.thousand_standard_cubic_feet_per_day, units.standard_cubic_feet_per_day): 1000,
            (units.thousand_standard_cubic_feet_per_day, units.cubic_meter_per_day): 1000 / 35.3146667,
            (units.thousand_standard_cubic_feet_per_day, units.cubic_meter_per_second): 1000 / 35.3146667 / 86_400,
        }
        # fmt: on

        try:
            return value * factor[(_from, _to)]
        except KeyError:
            UnitConverter._raise_cannot_convert_error(_from, _to)

    @staticmethod
    def _convert_volume(
        value: Union[int, float, np.ndarray, pd.Series],
        _from: Unit,
        _to: Unit,
    ) -> Union[int, float, np.ndarray, pd.Series]:
        """
        Convert volume value from '_from' to '_to'.
        This method is private and should not be called directly.
        """
        if value is None or _from == _to:
            return value

        # fmt: off
        # Syntax: (_from, _to): factor
        factor = {
            # From cubic_meter
            (units.cubic_meter, units.barrel): 6.289814,
            (units.cubic_meter, units.thousand_cubic_meter): 1 / 1000,
            (units.cubic_meter, units.million_cubic_meter): 1 / 1_000_000,
            (units.cubic_meter, units.standard_cubic_feet): 35.3146667,
            (units.cubic_meter, units.thousand_standard_cubic_feet): 35.3146667 / 1000,
            (units.cubic_meter, units.million_standard_cubic_feet): 35.3146667 / 1_000_000,
            (units.thousand_cubic_meter, units.cubic_meter): 1000,
            (units.million_cubic_meter, units.cubic_meter): 1_000_000,

            # From barrel
            (units.barrel, units.cubic_meter): 1 / 6.289814,

            # From standard_cubic_feet
            (units.standard_cubic_feet, units.thousand_standard_cubic_feet): 1 / 1000,
            (units.standard_cubic_feet, units.million_standard_cubic_feet): 1 / 1_000_000,
            (units.standard_cubic_feet, units.cubic_meter): 1 / 35.3146667,
            (units.thousand_standard_cubic_feet, units.standard_cubic_feet): 1000,
            (units.thousand_standard_cubic_feet, units.cubic_meter): 1000 / 35.3146667,
        }
        # fmt: on

        try:
            return value * factor[(_from, _to)]
        except KeyError:
            UnitConverter._raise_cannot_convert_error(_from, _to)

    @staticmethod
    def _convert_energy(
        value: Union[int, float, np.ndarray, pd.Series],
        _from: Unit,
        _to: Unit,
    ) -> Union[int, float, np.ndarray, pd.Series]:
        """
        Convert energy value from '_from' to '_to'.
        This method is private and should not be called directly.
        """
        if value is None or _from == _to:
            return value

        # fmt: off
        # Syntax: (_from, _to): factor
        factor = {
            # From kilo_calorie
            (units.kilo_calorie, units.british_thermal_unit): 4186.8 / 1055.05585262,
            (units.kilo_calorie, units.million_british_thermal_unit): 4186.8 / 1055.05585262 / 1_000_000,

            # From british_thermal_unit
            (units.british_thermal_unit, units.kilo_calorie): 453.59237 * 5/9 / 1000,
            (units.british_thermal_unit, units.million_british_thermal_unit): 1 / 1_000_000,
            (units.million_british_thermal_unit, units.kilo_calorie): 1_000_000 * 453.59237 * 5/9 / 1000,
        }
        # fmt: on

        try:
            return value * factor[(_from, _to)]
        except KeyError:
            UnitConverter._raise_cannot_convert_error(_from, _to)

    @staticmethod
    def _convert_area(
        value: Union[int, float, np.ndarray, pd.Series],
        _from: Unit,
        _to: Unit,
    ) -> Union[int, float, np.ndarray, pd.Series]:
        """
        Convert area value from '_from' to '_to'.
        This method is private and should not be called directly.
        """
        if value is None or _from == _to:
            return value

        # fmt: off
        # Syntax: (_from, _to): factor
        factor = {
            # From square_meter
            (units.square_meter, units.square_kilometer): 1 / 1_000_000,
            (units.square_meter, units.acre): 1 / 4046.85642,

            # From acre
            (units.acre, units.square_meter): 4046.85642,
        }
        # fmt: on

        try:
            return value * factor[(_from, _to)]
        except KeyError:
            UnitConverter._raise_cannot_convert_error(_from, _to)

    @staticmethod
    def _convert_volume_ratio(
        value: Union[int, float, np.ndarray, pd.Series],
        _from: Unit,
        _to: Unit,
    ) -> Union[int, float, np.ndarray, pd.Series]:
        """
        Convert volume ratio value from '_from' to '_to'.
        This method is private and should not be called directly.
        """
        if value is None or _from == _to:
            return value

        # fmt: off
        # Syntax: (_from, _to): factor
        factor = {
            # From barrel_per_stock_tank_barrel
            (units.barrel_per_stock_tank_barrel, units.cubic_meter_per_cubic_meter): 1,
            (units.barrel_per_stock_tank_barrel, units.standard_cubic_feet_per_stock_tank_barrel): 5.61458333,

            # From cubic_meter_per_cubic_meter
            (units.cubic_meter_per_cubic_meter, units.standard_cubic_feet_per_stock_tank_barrel): 5.61458333,

            # From standard_cubic_feet_per_stock_tank_barrel
            (units.standard_cubic_feet_per_stock_tank_barrel, units.cubic_meter_per_cubic_meter): 1 / 5.61458333
        }
        # fmt: on

        try:
            return value * factor[(_from, _to)]
        except KeyError:
            UnitConverter._raise_cannot_convert_error(_from, _to)
