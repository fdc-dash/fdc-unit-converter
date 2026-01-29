class Magnitude:
    """
    Representation of magnitude of units of measure.
    """

    # Define magnitudes
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
