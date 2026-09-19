r"""Waiting-based fuel and CO2 PROXY for simulation results.

*** SIMULATION PROXY / ESTIMATE. NOT A MEASUREMENT. NOT A CALIBRATED FUEL-CONSUMPTION MODEL. ***

The simulator has queues, waiting vehicle-seconds and vehicle counts. It has no engine type, vehicle
class, speed, acceleration, road length, engine load or measured idle rate. So the only environmental
quantity it can support is a proxy for fuel burned while vehicles WAIT in a queue::

    fuel_liters = waiting_vehicle_seconds / 3600  *  idle_fuel_rate_lph
    co2_kg      = fuel_liters  *  emission_factor_kg_per_liter

``waiting_vehicle_seconds`` is the simulator's own ``Metrics.total_waiting_time`` (queue occupancy
integrated over time, see ``metrics.py``). Both formulas are linear, so fuel and CO2 carry no
information beyond waiting time: a change in waiting time of x% is a change of x% in the proxy, for ANY
non-zero coefficients. The coefficients only set the absolute scale.

Coefficients: name, unit, meaning, provenance
---------------------------------------------
``emission_factor_kg_per_liter`` [kg CO2 per litre of fuel] - SOURCE-DERIVED.
    U.S. EPA, "Greenhouse Gas Emissions from a Typical Passenger Vehicle",
    https://www.epa.gov/greenvehicles/greenhouse-gas-emissions-typical-passenger-vehicle
    (page read for this project; page shows "last updated June 3, 2026"):
    "CO2 emissions from a gallon of gasoline: 8,887 grams CO2/gallon" and
    "CO2 emissions from a gallon of diesel: 10,180 grams CO2/gallon" (complete oxidation of the fuel's
    carbon). Converted with 1 US gallon = 3.785411784 L:
    gasoline = 8.887 / 3.785411784 = 2.3478 kg/L, diesel = 10.180 / 3.785411784 = 2.6893 kg/L.
    Default here: gasoline. This is a tailpipe (combustion) factor, not a life-cycle factor.

``idle_fuel_rate_lph`` [litres of fuel per vehicle-hour of waiting] - CONFIGURABLE SIMULATION PARAMETER.
    The environmental model uses a configurable waiting/idling fuel-rate proxy. The default is based on
    published passenger-vehicle idling examples and is not a calibrated fleet-average value. It is not a
    universal vehicle characteristic: engine size, fuel, accessories, temperature and driving pattern all
    change it.
    Default 0.6 L/h: 0.16 US gal/h x 3.78541 L/gal = 0.606 L/h, rounded. That is the lowest of the
    published examples below, so the proxy does not overstate fuel use.
    Published examples: Argonne National Laboratory, "Idling Reduction Savings Calculator",
    https://www.anl.gov/sites/www/files/2018-02/idling_worksheet.pdf . Passenger cars, no load:
    Ford Focus (gasoline) 0.16 gal/h = 0.61 L/h; Volkswagen Jetta (diesel) 0.17 gal/h = 0.64 L/h;
    Ford Crown Victoria (gasoline) 0.39 gal/h = 1.48 L/h. The U.S. Department of Energy gives the broader
    general statement that idling can use roughly 0.25-0.5 gal/h (0.95-1.9 L/h).
    Provenance of these figures: the worksheet's title, publisher and URL were confirmed, and the figures
    were supplied to this project by its owner from that worksheet. The tooling used to build the project
    could not retrieve the PDF text (HTTP 403), so the figures were not independently re-read there.
    The sensitivity values (0.6, 0.9, 1.2, 1.9 L/h, about 0.16-0.5 gal/h) are scenario parameters that
    span these examples and the DOE range. They are not vehicle characteristics.

What the proxy does NOT model
-----------------------------
acceleration/deceleration, vehicle speed, vehicle type, engine efficiency, road gradient, temperature,
congestion-dependent fuel burn, actual vehicle trajectories, exhaust composition, NOx/PM emissions,
fuel used while moving, and fuel of vehicles in transit between intersections (they are not "waiting"
in the simulator). It estimates only a waiting/idling-related quantity.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

PROXY_LABEL = "SIMULATION PROXY (waiting/idling only) - estimate, not a measurement"

SECONDS_PER_HOUR = 3600.0
LITERS_PER_US_GALLON = 3.785411784  # exact by definition (231 cubic inches)

# Source-derived (U.S. EPA, see module docstring): grams of CO2 per US gallon of fuel burned
EPA_GASOLINE_G_CO2_PER_US_GALLON = 8887.0
EPA_DIESEL_G_CO2_PER_US_GALLON = 10180.0
GASOLINE_KG_CO2_PER_LITER = EPA_GASOLINE_G_CO2_PER_US_GALLON / 1000.0 / LITERS_PER_US_GALLON
DIESEL_KG_CO2_PER_LITER = EPA_DIESEL_G_CO2_PER_US_GALLON / 1000.0 / LITERS_PER_US_GALLON

# Configurable scenario parameter (see module docstring): 0.16 gal/h x 3.78541 L/gal = 0.606 L/h, rounded
DEFAULT_IDLE_FUEL_RATE_LPH = 0.6
SENSITIVITY_IDLE_FUEL_RATES_LPH = (0.6, 0.9, 1.2, 1.9)  # scenario parameters, ~0.16 .. ~0.5 US gal/h

# Published passenger-vehicle idling examples, US gal/h, no load. Argonne National Laboratory, "Idling
# Reduction Savings Calculator" (figures as supplied by the project owner; see the module docstring).
ARGONNE_IDLE_EXAMPLES_GAL_PER_HOUR = {
    "Ford Focus (gasoline)": 0.16,
    "Volkswagen Jetta (diesel)": 0.17,
    "Ford Crown Victoria (gasoline)": 0.39,
}
DOE_GENERAL_IDLE_RANGE_GAL_PER_HOUR = (0.25, 0.5)  # broader general statement (U.S. DOE)


def gallons_per_hour_to_liters_per_hour(gallons_per_hour: float) -> float:
    """US gallons per hour -> litres per hour."""
    return gallons_per_hour * LITERS_PER_US_GALLON


def _check_nonneg_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class EnvironmentalConfig:
    """Coefficients of the waiting-based proxy. Every field has a unit in its name and can be overridden."""

    idle_fuel_rate_lph: float = DEFAULT_IDLE_FUEL_RATE_LPH  # L of fuel per vehicle-hour waiting (configurable scenario parameter)
    emission_factor_kg_per_liter: float = GASOLINE_KG_CO2_PER_LITER  # kg CO2 per L fuel (EPA, gasoline)
    fuel_type: str = "gasoline"  # label only

    def __post_init__(self) -> None:
        _check_nonneg_finite("idle_fuel_rate_lph", self.idle_fuel_rate_lph)
        _check_nonneg_finite("emission_factor_kg_per_liter", self.emission_factor_kg_per_liter)

    def provenance(self) -> dict:
        return {
            "label": PROXY_LABEL,
            "idle_fuel_rate_lph": self.idle_fuel_rate_lph,
            "idle_fuel_rate_status": "CONFIGURABLE SIMULATION PARAMETER (not a universal vehicle constant)",
            "idle_fuel_rate_basis": "Default 0.6 L/h = 0.16 gal/h x 3.78541 L/gal = 0.606 L/h, based on published "
                                    "passenger-vehicle idling examples (Argonne National Laboratory 'Idling Reduction "
                                    "Savings Calculator', no load: Ford Focus gasoline 0.16, Volkswagen Jetta diesel "
                                    "0.17, Ford Crown Victoria gasoline 0.39 gal/h; U.S. DOE general statement about "
                                    "0.25-0.5 gal/h). Not a calibrated fleet-average value.",
            "emission_factor_kg_per_liter": self.emission_factor_kg_per_liter,
            "emission_factor_basis": "SOURCE-DERIVED: U.S. EPA 8,887 g CO2 per US gallon gasoline "
                                     "(10,180 for diesel), converted at 3.785411784 L per gallon",
            "fuel_type": self.fuel_type,
        }


def waiting_hours(waiting_vehicle_seconds: float) -> float:
    """Vehicle-seconds -> vehicle-hours."""
    return _check_nonneg_finite("waiting_vehicle_seconds", waiting_vehicle_seconds) / SECONDS_PER_HOUR


def fuel_liters(waiting_vehicle_seconds: float, config: EnvironmentalConfig | None = None) -> float:
    """Fuel proxy: ``waiting_vehicle_seconds / 3600 * idle_fuel_rate_lph`` [litres]."""
    cfg = config or EnvironmentalConfig()
    return waiting_hours(waiting_vehicle_seconds) * cfg.idle_fuel_rate_lph


def co2_kg(fuel_liters_value: float, config: EnvironmentalConfig | None = None) -> float:
    """CO2 proxy: ``fuel_liters * emission_factor_kg_per_liter`` [kg]."""
    cfg = config or EnvironmentalConfig()
    return _check_nonneg_finite("fuel_liters", fuel_liters_value) * cfg.emission_factor_kg_per_liter


@dataclass(frozen=True)
class EnvironmentalEstimate:
    waiting_vehicle_seconds: float  # simulator output (measured in the simulation)
    fuel_liters: float  # PROXY
    co2_kg: float  # PROXY
    config: EnvironmentalConfig
    label: str = PROXY_LABEL

    def per_vehicle(self, admitted_vehicles: float) -> tuple[float, float]:
        """(fuel litres, CO2 kg) per admitted vehicle; (0, 0) when nobody was admitted."""
        if admitted_vehicles < 0 or not math.isfinite(admitted_vehicles):
            raise ValueError("admitted_vehicles must be finite and >= 0")
        if admitted_vehicles == 0:
            return 0.0, 0.0
        return self.fuel_liters / admitted_vehicles, self.co2_kg / admitted_vehicles

    def as_dict(self) -> dict:
        return {**asdict(self.config), "waiting_vehicle_seconds": self.waiting_vehicle_seconds,
                "fuel_liters_proxy": self.fuel_liters, "co2_kg_proxy": self.co2_kg, "label": self.label}


def estimate(waiting_vehicle_seconds: float, config: EnvironmentalConfig | None = None) -> EnvironmentalEstimate:
    cfg = config or EnvironmentalConfig()
    fuel = fuel_liters(waiting_vehicle_seconds, cfg)
    return EnvironmentalEstimate(float(waiting_vehicle_seconds), fuel, co2_kg(fuel, cfg), cfg)


def percent_change(baseline: float, value: float) -> float | None:
    """Signed % change of ``value`` relative to ``baseline``; None when the baseline is 0."""
    return None if baseline == 0 else (value - baseline) / abs(baseline) * 100.0


def sensitivity_configs(
    idle_fuel_rates_lph: Iterable[float] = SENSITIVITY_IDLE_FUEL_RATES_LPH,
    emission_factors: Mapping[str, float] | None = None,
) -> list[EnvironmentalConfig]:
    """Grid of proxy configurations: every idle rate x every emission factor (default gasoline, diesel)."""
    factors = emission_factors or {"gasoline": GASOLINE_KG_CO2_PER_LITER, "diesel": DIESEL_KG_CO2_PER_LITER}
    return [EnvironmentalConfig(rate, ef, fuel_type=name)
            for rate in idle_fuel_rates_lph for name, ef in factors.items()]
