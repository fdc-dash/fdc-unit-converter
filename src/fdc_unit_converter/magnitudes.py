class Magnitude:
    """
    Representation of magnitude of units of measure.

    Two units can only be converted into each other when they declare the same
    magnitude. Every magnitude has one base unit, and every unit of that
    magnitude is defined relative to it (see ``units.py``).
    """

    AREA = "area"
    CALORIFIC_VALUE = "calorific_value"
    COMPRESSIBILITY = "compressibility"
    CURRENCY = "currency"
    DENSITY = "density"
    ELECTRIC_CURRENT = "electric_current"
    ENERGY = "energy"
    FORCE = "force"
    FREQUENCY = "frequency"
    GAS_GRAVITY = "gas_gravity"
    LENGTH = "length"
    LIQUID_GRAVITY = "liquid_gravity"
    MASS = "mass"
    # Mass of solute per mass of solution (ppm, ppb, % by mass).
    MASS_FRACTION = "mass_fraction"
    # Mass of solute per volume of solution (mg/L, g/L, pg/mL).
    #
    # Deliberately a different magnitude from MASS_FRACTION: converting between
    # the two depends on the density of the solution, so there is no fixed
    # factor. For brines (density 1.05-1.20) assuming one would introduce a
    # 5-20% error. Convert the density explicitly in caller code instead.
    MASS_CONCENTRATION = "mass_concentration"
    MOLAR_MASS = "molar_mass"
    PERMEABILITY = "permeability"
    PRESSURE = "pressure"
    RATE = "rate"
    TEMPERATURE = "temperature"
    TIME = "time"
    VISCOSITY = "viscosity"
    VOLUME = "volume"
    VOLUME_RATIO = "volume_ratio"

    # This registry holds units for each magnitude.
    # It is populated when Unit instances are created.
    # Key: magnitude (str), Value: list of Unit instances
    _units_registry: dict = {}

    @staticmethod
    def list_magnitudes() -> list:
        """Return all available magnitudes."""
        return list(Magnitude._units_registry.keys())

    @classmethod
    def list_magnitude_units(cls, magnitude: str) -> list:
        """Return all units for a given magnitude."""
        return cls._units_registry.get(magnitude, [])
