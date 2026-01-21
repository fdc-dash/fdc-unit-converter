class Magnitude:
    """
    Representation of magnitude of units of measure.
    """

    AREA = "area"
    COMPRESSIBILITY = "compressibility"
    CURRENCY = "currency"
    DENSITY = "density"
    ENERGY = "energy"
    GAS_GRAVITY = "gas_gravity"
    LENGTH = "length"
    LIQUID_GRAVITY = "liquid_gravity"
    MASS = "mass"
    PERMEABILITY = "permeability"
    PRESSURE = "pressure"
    RATE = "rate"
    TEMPERATURE = "temperature"
    TIME = "time"
    VISCOSITY = "viscosity"
    VOLUME = "volume"
    VOLUME_RATIO = "volume_ratio"


class Unit:
    """
    Representation of units of measure.
    """

    def __init__(self, name: str, symbol: str, magnitude: str):
        self.name = name
        self.symbol = symbol
        self.magnitude = magnitude


# fmt: off

# Area
square_meter = Unit("square meter", "m2", Magnitude.AREA)
square_kilometer = Unit("square kilometer", "km2", Magnitude.AREA)
acre = Unit("acre", "acre", Magnitude.AREA)

# Compressibility
inverse_pascal = Unit("inverse pascal", "1/Pa", Magnitude.COMPRESSIBILITY)
inverse_bar = Unit("inverse bar", "1/bar", Magnitude.COMPRESSIBILITY)
inverse_kilogram_per_square_centimeter = Unit("inverse kilogram per square centimeter", "1/(kg/cm2)", Magnitude.COMPRESSIBILITY)  # noqa: E501
inverse_pound_per_square_inch = Unit("inverse pound per square inch", "1/psi", Magnitude.COMPRESSIBILITY)

# Currency
dollar = Unit("dollar", "USD", Magnitude.CURRENCY)
thousand_dollar = Unit("thousand dollar", "MUSD", Magnitude.CURRENCY)
million_dollar = Unit("million dollar", "MMUSD", Magnitude.CURRENCY)

# Density
kilogram_per_cubic_meter = Unit("kilogram per cubic meter", "kg/m3", Magnitude.DENSITY)
gram_per_cubic_centimeter = Unit("gram per cubic centimeter", "g/cm3", Magnitude.DENSITY)
pound_per_cubic_foot = Unit("pound per cubic foot", "lb/ft3", Magnitude.DENSITY)

# Energy
british_thermal_unit = Unit("british thermal unit", "Btu", Magnitude.ENERGY)
million_british_thermal_unit = Unit("million british thermal unit", "MMBtu", Magnitude.ENERGY)
kilo_calorie = Unit("kilo calorie", "kcal", Magnitude.ENERGY)

# Length
millimeter = Unit("millimeter", "mm", Magnitude.LENGTH)
meter = Unit("meter", "m", Magnitude.LENGTH)
kilometer = Unit("kilometer", "km", Magnitude.LENGTH)
foot = Unit("foot", "ft", Magnitude.LENGTH)
inch = Unit("inch", "in", Magnitude.LENGTH)

# Liquid gravity
API_gravity = Unit("API gravity", "°API", Magnitude.LIQUID_GRAVITY)
specific_gravity = Unit("specific gravity", "SG", Magnitude.LIQUID_GRAVITY)

# Permeability
nanodarcy = Unit("nanodarcy", "nd", Magnitude.PERMEABILITY)
millidarcy = Unit("millidarcy", "md", Magnitude.PERMEABILITY)
darcy = Unit("darcy", "d", Magnitude.PERMEABILITY)
microdarcy = Unit("microdarcy", "μd", Magnitude.PERMEABILITY)
square_meter_permeability = Unit("square meter", "m2", Magnitude.PERMEABILITY)

# Pressure
bar = Unit("bar", "bar", Magnitude.PRESSURE)
pascal = Unit("pascal", "Pa", Magnitude.PRESSURE)
kilopascal = Unit("kilopascal", "kPa", Magnitude.PRESSURE)
pound_per_square_inch = Unit("pound per square inch", "psi", Magnitude.PRESSURE)
kilogram_per_square_centimeter = Unit("kilogram per square centimeter", "kg/cm2", Magnitude.PRESSURE)

# Rate
cubic_meter_per_day = Unit("cubic meter per day", "m3/day", Magnitude.RATE)
cubic_meter_per_second = Unit("cubic meter per second", "m3/s", Magnitude.RATE)
thousand_cubic_meter_per_day = Unit("thousand cubic meter per day", "Mm3/day", Magnitude.RATE)
stock_tank_barrel_per_day = Unit("stock tank barrel per day", "STB/day", Magnitude.RATE)
standard_cubic_feet_per_day = Unit("standard cubic feet per day", "scf/day", Magnitude.RATE)
thousand_standard_cubic_feet_per_day = Unit("thousand standard cubic feet per day", "Mscf/day", Magnitude.RATE)

# Temperature
celsius = Unit("celsius", "°C", Magnitude.TEMPERATURE)
fahrenheit = Unit("fahrenheit", "°F", Magnitude.TEMPERATURE)
rankine = Unit("rankine", "°R", Magnitude.TEMPERATURE)
kelvin = Unit("kelvin", "K", Magnitude.TEMPERATURE)

# Time
second = Unit("second", "s", Magnitude.TIME)
minute = Unit("minute", "min", Magnitude.TIME)
hour = Unit("hour", "hr", Magnitude.TIME)
day = Unit("day", "day", Magnitude.TIME)

# Viscosity
centipoise = Unit("centipoise", "cP", Magnitude.VISCOSITY)
poise = Unit("poise", "P", Magnitude.VISCOSITY)

# Volume
cubic_meter = Unit("cubic meter", "m3", Magnitude.VOLUME)
thousand_cubic_meter = Unit("thousand cubic meter", "Mm3", Magnitude.VOLUME)
million_cubic_meter = Unit("million cubic meter", "MMm3", Magnitude.VOLUME)
standard_cubic_feet = Unit("standard cubic feet", "scf", Magnitude.VOLUME)
thousand_standard_cubic_feet = Unit("thousand standard cubic feet", "Mscf", Magnitude.VOLUME)
million_standard_cubic_feet = Unit("million standard cubic feet", "MMscf", Magnitude.VOLUME)
barrel = Unit("barrel", "bbl", Magnitude.VOLUME)

# Volume ratio
barrel_per_stock_tank_barrel = Unit("barrel per stock tank barrel", "bbl/STB", Magnitude.VOLUME_RATIO)
cubic_meter_per_cubic_meter = Unit("cubic meter per cubic meter", "m3/m3", Magnitude.VOLUME_RATIO)
standard_cubic_feet_per_stock_tank_barrel = Unit("standard cubic feet per stock tank barrel", "scf/STB", Magnitude.VOLUME_RATIO)  # noqa: E501

# fmt: on
