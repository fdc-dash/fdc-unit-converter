from typing import List, Union

import numpy as np
import pandas as pd

from .units import Unit


class UnitConverter:
    """
    Convert value from one unit to another.

    Conversions go through the base unit of the magnitude rather than through a
    table of declared pairs, so every ordered pair of units within a magnitude
    converts, the inverse is always the exact reverse, and the result does not
    depend on the path taken.
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

        if _from is _to:
            return value

        if not (_from.is_convertible and _to.is_convertible):
            UnitConverter._raise_cannot_convert_error(_from, _to)

        return _to.from_base(_from.to_base(value))

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
        ADMITED_TYPES = ADMITED_MEMBER_TYPES + (list, np.ndarray, pd.Series)

        if not isinstance(value, ADMITED_TYPES):
            raise ValueError("Cannot convert non-numeric values")

        if isinstance(value, list):
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
