"""
Units of measure, each defined relative to the base unit of its magnitude.

Every unit declares how to reach its magnitude's base unit:

    base_value = (value + offset) * factor

where ``offset`` is expressed in this unit's own scale -- how far this unit's
zero sits from the base unit's zero -- so that temperature reads as the
familiar ``(F - 32) * 5/9``. Every magnitude other than temperature has
``offset = 0``, which reduces the expression to ``value * factor``.

The inverse is derived from the same two numbers, so a conversion between
any two units of a magnitude is ``B.from_base(A.to_base(value))`` and no pair
has to be declared. Adding a unit is one line; its conversions to and from
every other unit of the magnitude come for free and cannot desynchronise.

A handful of relations are not affine -- API gravity is reciprocal, not a
factor plus an offset -- so a unit may instead supply an explicit
``to_base``/``from_base`` pair. See ``API_gravity`` below.
"""

from typing import Callable, Optional

from .magnitudes import Magnitude

__all__ = ["Magnitude", "Unit"]


class Unit:
    """
    Representation of units of measure.

    Args:
        name: Human readable name, used in error messages.
        symbol: Short symbol, e.g. ``"m3/day"``.
        magnitude: One of the :class:`Magnitude` constants. Only units sharing
            a magnitude can be converted into each other.
        factor: Multiplier that takes a value of this unit to the base unit of
            its magnitude.
        offset: Constant added *before* applying ``factor``, expressed in this
            unit's own scale. Only temperature needs it.
        to_base: Optional explicit conversion to the base unit, for relations
            that are not affine. Overrides ``factor`` and ``offset``.
        from_base: Inverse of ``to_base``. Required whenever ``to_base`` is
            given.
    """

    def __init__(
        self,
        name: str,
        symbol: str,
        magnitude: str,
        factor: float = 1.0,
        offset: float = 0.0,
        to_base: Optional[Callable] = None,
        from_base: Optional[Callable] = None,
    ):
        if (to_base is None) != (from_base is None):
            raise ValueError(f"unit '{name}' must define both 'to_base' and 'from_base', or neither")

        self.name = name
        self.symbol = symbol
        self.magnitude = magnitude
        self.factor = factor
        self.offset = offset
        self._to_base = to_base
        self._from_base = from_base

        # Register the unit in the registry for the given magnitude
        Magnitude._units_registry.setdefault(magnitude, []).append(self)

    @property
    def is_convertible(self) -> bool:
        """Whether this unit carries enough information to be converted."""
        return self._to_base is not None or self.factor != 0

    def to_base(self, value):
        """Express ``value`` in the base unit of this unit's magnitude."""
        if self._to_base is not None:
            return self._to_base(value)
        if self.offset:
            return (value + self.offset) * self.factor
        return value * self.factor

    def from_base(self, value):
        """Express ``value``, given in the base unit of this magnitude, in this unit."""
        if self._from_base is not None:
            return self._from_base(value)
        if self.offset:
            return value / self.factor - self.offset
        return value / self.factor

    def __repr__(self) -> str:
        return f"Unit({self.name!r}, {self.symbol!r}, {self.magnitude!r})"


# fmt: off

# ------------------------------------------------------------------------------
# Shared constants
#
# These are the conversion factors the pairwise tables used before, kept as
# named constants so the derived units reproduce the historical values exactly.
# ------------------------------------------------------------------------------
METER_PER_FOOT = 0.3048                     # exact, international foot
METER_PER_INCH = 0.0254                     # exact, international inch
SQUARE_METER_PER_ACRE = 4046.85642
CUBIC_METER_PER_BARREL = 1 / 6.289814
CUBIC_FEET_PER_CUBIC_METER = 35.3146667
STANDARD_CUBIC_FEET_PER_STOCK_TANK_BARREL = 5.61458333
PASCAL_PER_BAR = 100_000
PASCAL_PER_KILOGRAM_PER_SQUARE_CENTIMETER = 98066.5
PASCAL_PER_POUND_PER_SQUARE_INCH = 6894.75729
SQUARE_METER_PER_DARCY = 9.869233e-13
JOULE_PER_KILO_CALORIE = 4186.8
JOULE_PER_BRITISH_THERMAL_UNIT = 1055.05585262
BRITISH_THERMAL_UNIT_PER_KILO_CALORIE = JOULE_PER_KILO_CALORIE / JOULE_PER_BRITISH_THERMAL_UNIT
KILO_CALORIE_PER_BRITISH_THERMAL_UNIT = JOULE_PER_BRITISH_THERMAL_UNIT / JOULE_PER_KILO_CALORIE
SECONDS_PER_DAY = 86_400
KILOGRAM_PER_POUND = 0.45359237
NEWTON_PER_KILOGRAM_FORCE = 9.80665

# ------------------------------------------------------------------------------
# Area -- base: square_meter
# ------------------------------------------------------------------------------
square_meter = Unit("square meter", "m2", Magnitude.AREA, 1)
square_kilometer = Unit("square kilometer", "km2", Magnitude.AREA, 1_000_000)
acre = Unit("acre", "acre", Magnitude.AREA, SQUARE_METER_PER_ACRE)

# ------------------------------------------------------------------------------
# Calorific value -- base: kilocalorie_per_cubic_meter
# ------------------------------------------------------------------------------
kilocalorie_per_cubic_meter = Unit("kilocalorie per cubic meter", "kcal/m3", Magnitude.CALORIFIC_VALUE, 1)
megajoule_per_cubic_meter = Unit("megajoule per cubic meter", "MJ/m3", Magnitude.CALORIFIC_VALUE, 1_000_000 / JOULE_PER_KILO_CALORIE)  # noqa: E501
british_thermal_unit_per_standard_cubic_feet = Unit("british thermal unit per standard cubic feet", "Btu/scf", Magnitude.CALORIFIC_VALUE, KILO_CALORIE_PER_BRITISH_THERMAL_UNIT * CUBIC_FEET_PER_CUBIC_METER)  # noqa: E501

# ------------------------------------------------------------------------------
# Compressibility -- base: inverse_pascal
#
# These units are inverses of pressure, so each factor is the reciprocal of the
# corresponding pressure factor.
# ------------------------------------------------------------------------------
inverse_pascal = Unit("inverse pascal", "1/Pa", Magnitude.COMPRESSIBILITY, 1)
inverse_bar = Unit("inverse bar", "1/bar", Magnitude.COMPRESSIBILITY, 1 / PASCAL_PER_BAR)
inverse_kilogram_per_square_centimeter = Unit("inverse kilogram per square centimeter", "1/(kg/cm2)", Magnitude.COMPRESSIBILITY, 1 / PASCAL_PER_KILOGRAM_PER_SQUARE_CENTIMETER)  # noqa: E501
inverse_pound_per_square_inch = Unit("inverse pound per square inch", "1/psi", Magnitude.COMPRESSIBILITY, 1 / PASCAL_PER_POUND_PER_SQUARE_INCH)  # noqa: E501

# ------------------------------------------------------------------------------
# Currency -- base: dollar
# ------------------------------------------------------------------------------
dollar = Unit("dollar", "USD", Magnitude.CURRENCY, 1)
thousand_dollar = Unit("thousand dollar", "MUSD", Magnitude.CURRENCY, 1_000)
million_dollar = Unit("million dollar", "MMUSD", Magnitude.CURRENCY, 1_000_000)

# ------------------------------------------------------------------------------
# Density -- base: kilogram_per_cubic_meter
# ------------------------------------------------------------------------------
kilogram_per_cubic_meter = Unit("kilogram per cubic meter", "kg/m3", Magnitude.DENSITY, 1)
gram_per_cubic_centimeter = Unit("gram per cubic centimeter", "g/cm3", Magnitude.DENSITY, 1_000)
pound_per_cubic_foot = Unit("pound per cubic foot", "lb/ft3", Magnitude.DENSITY, KILOGRAM_PER_POUND / METER_PER_FOOT ** 3)  # noqa: E501

# ------------------------------------------------------------------------------
# Electric current -- base: ampere
# ------------------------------------------------------------------------------
ampere = Unit("ampere", "A", Magnitude.ELECTRIC_CURRENT, 1)
milliampere = Unit("milliampere", "mA", Magnitude.ELECTRIC_CURRENT, 1 / 1_000)
kiloampere = Unit("kiloampere", "kA", Magnitude.ELECTRIC_CURRENT, 1_000)

# ------------------------------------------------------------------------------
# Energy -- base: british_thermal_unit
# ------------------------------------------------------------------------------
british_thermal_unit = Unit("british thermal unit", "Btu", Magnitude.ENERGY, 1)
million_british_thermal_unit = Unit("million british thermal unit", "MMBtu", Magnitude.ENERGY, 1_000_000)
kilo_calorie = Unit("kilo calorie", "kcal", Magnitude.ENERGY, BRITISH_THERMAL_UNIT_PER_KILO_CALORIE)

# ------------------------------------------------------------------------------
# Force -- base: newton
# ------------------------------------------------------------------------------
newton = Unit("newton", "N", Magnitude.FORCE, 1)
kilonewton = Unit("kilonewton", "kN", Magnitude.FORCE, 1_000)
kilogram_force = Unit("kilogram force", "kgf", Magnitude.FORCE, NEWTON_PER_KILOGRAM_FORCE)
pound_force = Unit("pound force", "lbf", Magnitude.FORCE, KILOGRAM_PER_POUND * NEWTON_PER_KILOGRAM_FORCE)

# ------------------------------------------------------------------------------
# Frequency -- base: hertz
# ------------------------------------------------------------------------------
hertz = Unit("hertz", "Hz", Magnitude.FREQUENCY, 1)
kilohertz = Unit("kilohertz", "kHz", Magnitude.FREQUENCY, 1_000)
revolution_per_minute = Unit("revolution per minute", "rpm", Magnitude.FREQUENCY, 1 / 60)

# ------------------------------------------------------------------------------
# Gas gravity -- base: gas_specific_gravity
#
# Dimensionless, relative to air. Single unit; declared so the magnitude is not
# empty and so gas gravity values can be typed like every other quantity.
# ------------------------------------------------------------------------------
gas_specific_gravity = Unit("gas specific gravity", "SG_gas", Magnitude.GAS_GRAVITY, 1)

# ------------------------------------------------------------------------------
# Length -- base: meter
# ------------------------------------------------------------------------------
meter = Unit("meter", "m", Magnitude.LENGTH, 1)
millimeter = Unit("millimeter", "mm", Magnitude.LENGTH, 1 / 1_000)
centimeter = Unit("centimeter", "cm", Magnitude.LENGTH, 1 / 100)
kilometer = Unit("kilometer", "km", Magnitude.LENGTH, 1_000)
foot = Unit("foot", "ft", Magnitude.LENGTH, METER_PER_FOOT)
inch = Unit("inch", "in", Magnitude.LENGTH, METER_PER_INCH)

# ------------------------------------------------------------------------------
# Liquid gravity -- base: specific_gravity
#
# API gravity is reciprocal in specific gravity, not affine, so it declares its
# own conversion pair instead of a factor and an offset.
# ------------------------------------------------------------------------------
specific_gravity = Unit("specific gravity", "SG", Magnitude.LIQUID_GRAVITY, 1)
API_gravity = Unit(
    "API gravity", "°API", Magnitude.LIQUID_GRAVITY,
    to_base=lambda api: 141.5 / (api + 131.5),
    from_base=lambda sg: (141.5 / sg) - 131.5,
)

# ------------------------------------------------------------------------------
# Mass -- base: kilogram
# ------------------------------------------------------------------------------
kilogram = Unit("kilogram", "kg", Magnitude.MASS, 1)
gram = Unit("gram", "g", Magnitude.MASS, 1 / 1_000)
tonne = Unit("tonne", "t", Magnitude.MASS, 1_000)
pound = Unit("pound", "lb", Magnitude.MASS, KILOGRAM_PER_POUND)

# ------------------------------------------------------------------------------
# Mass concentration (mass of solute per volume of solution) -- base: milligram_per_liter
#
# Kept separate from MASS_FRACTION on purpose: ppm is mass/mass and mg/L is
# mass/volume, so converting between them requires the density of the solution.
# ------------------------------------------------------------------------------
milligram_per_liter = Unit("milligram per liter", "mg/L", Magnitude.MASS_CONCENTRATION, 1)
microgram_per_liter = Unit("microgram per liter", "ug/L", Magnitude.MASS_CONCENTRATION, 1 / 1_000)
gram_per_liter = Unit("gram per liter", "g/L", Magnitude.MASS_CONCENTRATION, 1_000)
picogram_per_milliliter = Unit("picogram per milliliter", "pg/mL", Magnitude.MASS_CONCENTRATION, 1 / 1_000_000)

# ------------------------------------------------------------------------------
# Mass fraction (mass of solute per mass of solution) -- base: part_per_million
# ------------------------------------------------------------------------------
part_per_million = Unit("part per million", "ppm", Magnitude.MASS_FRACTION, 1)
part_per_billion = Unit("part per billion", "ppb", Magnitude.MASS_FRACTION, 1 / 1_000)
percent_by_mass = Unit("percent by mass", "%wt", Magnitude.MASS_FRACTION, 10_000)

# ------------------------------------------------------------------------------
# Molar mass -- base: kilogram_per_kilomole
#
# kg/kmol and g/mol are numerically identical; both are declared because both
# are used in chromatography reports.
# ------------------------------------------------------------------------------
kilogram_per_kilomole = Unit("kilogram per kilomole", "kg/kmol", Magnitude.MOLAR_MASS, 1)
gram_per_mole = Unit("gram per mole", "g/mol", Magnitude.MOLAR_MASS, 1)

# ------------------------------------------------------------------------------
# Permeability -- base: darcy
# ------------------------------------------------------------------------------
darcy = Unit("darcy", "d", Magnitude.PERMEABILITY, 1)
millidarcy = Unit("millidarcy", "md", Magnitude.PERMEABILITY, 1 / 1_000)
microdarcy = Unit("microdarcy", "μd", Magnitude.PERMEABILITY, 1 / 1_000_000)
nanodarcy = Unit("nanodarcy", "nd", Magnitude.PERMEABILITY, 1 / 1_000_000_000)
square_meter_permeability = Unit("square meter", "m2", Magnitude.PERMEABILITY, 1 / SQUARE_METER_PER_DARCY)

# ------------------------------------------------------------------------------
# Pressure -- base: pascal
# ------------------------------------------------------------------------------
pascal = Unit("pascal", "Pa", Magnitude.PRESSURE, 1)
kilopascal = Unit("kilopascal", "kPa", Magnitude.PRESSURE, 1_000)
bar = Unit("bar", "bar", Magnitude.PRESSURE, PASCAL_PER_BAR)
kilogram_per_square_centimeter = Unit("kilogram per square centimeter", "kg/cm2", Magnitude.PRESSURE, PASCAL_PER_KILOGRAM_PER_SQUARE_CENTIMETER)  # noqa: E501
pound_per_square_inch = Unit("pound per square inch", "psi", Magnitude.PRESSURE, PASCAL_PER_POUND_PER_SQUARE_INCH)

# ------------------------------------------------------------------------------
# Rate -- base: cubic_meter_per_day
# ------------------------------------------------------------------------------
cubic_meter_per_day = Unit("cubic meter per day", "m3/day", Magnitude.RATE, 1)
thousand_cubic_meter_per_day = Unit("thousand cubic meter per day", "Mm3/day", Magnitude.RATE, 1_000)
cubic_meter_per_second = Unit("cubic meter per second", "m3/s", Magnitude.RATE, SECONDS_PER_DAY)
liter_per_day = Unit("liter per day", "L/day", Magnitude.RATE, 1 / 1_000)
stock_tank_barrel_per_day = Unit("stock tank barrel per day", "STB/day", Magnitude.RATE, CUBIC_METER_PER_BARREL)
standard_cubic_feet_per_day = Unit("standard cubic feet per day", "scf/day", Magnitude.RATE, 1 / CUBIC_FEET_PER_CUBIC_METER)  # noqa: E501
thousand_standard_cubic_feet_per_day = Unit("thousand standard cubic feet per day", "Mscf/day", Magnitude.RATE, 1_000 / CUBIC_FEET_PER_CUBIC_METER)  # noqa: E501

# ------------------------------------------------------------------------------
# Temperature -- base: celsius
#
# The only magnitude that needs an offset. The inverse is not 1/factor; it is
# value / factor - offset, which Unit.from_base derives from the same numbers.
# Celsius is the base rather than kelvin because every historical conversion in
# this magnitude is anchored on it, so this choice reproduces them bit for bit.
# ------------------------------------------------------------------------------
celsius = Unit("celsius", "°C", Magnitude.TEMPERATURE, 1, 0)
kelvin = Unit("kelvin", "K", Magnitude.TEMPERATURE, 1, -273.15)
fahrenheit = Unit("fahrenheit", "°F", Magnitude.TEMPERATURE, 5 / 9, -32)
rankine = Unit("rankine", "°R", Magnitude.TEMPERATURE, 5 / 9, -491.67)

# ------------------------------------------------------------------------------
# Time -- base: second
# ------------------------------------------------------------------------------
second = Unit("second", "s", Magnitude.TIME, 1)
minute = Unit("minute", "min", Magnitude.TIME, 60)
hour = Unit("hour", "hr", Magnitude.TIME, 3_600)
day = Unit("day", "day", Magnitude.TIME, SECONDS_PER_DAY)

# ------------------------------------------------------------------------------
# Viscosity -- base: pascal_second
# ------------------------------------------------------------------------------
pascal_second = Unit("pascal second", "Pa.s", Magnitude.VISCOSITY, 1)
millipascal_second = Unit("millipascal second", "mPa.s", Magnitude.VISCOSITY, 1 / 1_000)
poise = Unit("poise", "P", Magnitude.VISCOSITY, 1 / 10)
centipoise = Unit("centipoise", "cP", Magnitude.VISCOSITY, 1 / 1_000)

# ------------------------------------------------------------------------------
# Volume -- base: cubic_meter
# ------------------------------------------------------------------------------
cubic_meter = Unit("cubic meter", "m3", Magnitude.VOLUME, 1)
thousand_cubic_meter = Unit("thousand cubic meter", "Mm3", Magnitude.VOLUME, 1_000)
million_cubic_meter = Unit("million cubic meter", "MMm3", Magnitude.VOLUME, 1_000_000)
liter = Unit("liter", "L", Magnitude.VOLUME, 1 / 1_000)
barrel = Unit("barrel", "bbl", Magnitude.VOLUME, CUBIC_METER_PER_BARREL)
standard_cubic_feet = Unit("standard cubic feet", "scf", Magnitude.VOLUME, 1 / CUBIC_FEET_PER_CUBIC_METER)
thousand_standard_cubic_feet = Unit("thousand standard cubic feet", "Mscf", Magnitude.VOLUME, 1_000 / CUBIC_FEET_PER_CUBIC_METER)  # noqa: E501
million_standard_cubic_feet = Unit("million standard cubic feet", "MMscf", Magnitude.VOLUME, 1_000_000 / CUBIC_FEET_PER_CUBIC_METER)  # noqa: E501

# ------------------------------------------------------------------------------
# Volume ratio -- base: cubic_meter_per_cubic_meter
# ------------------------------------------------------------------------------
cubic_meter_per_cubic_meter = Unit("cubic meter per cubic meter", "m3/m3", Magnitude.VOLUME_RATIO, 1)
barrel_per_stock_tank_barrel = Unit("barrel per stock tank barrel", "bbl/STB", Magnitude.VOLUME_RATIO, 1)
standard_cubic_feet_per_stock_tank_barrel = Unit("standard cubic feet per stock tank barrel", "scf/STB", Magnitude.VOLUME_RATIO, 1 / STANDARD_CUBIC_FEET_PER_STOCK_TANK_BARREL)  # noqa: E501

# fmt: on
