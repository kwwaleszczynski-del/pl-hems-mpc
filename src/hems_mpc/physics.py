"""Rooftop PV model, heat pump COP curve, plant parameters and synthetic household load profiles."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProsumerPlantSpec:
    """Parameters of a Polish residential prosumer site (PV + LFP BESS + Heat Pump + EV)."""
    pv_kwp: float = 10.0
    pv_inverter_kw: float = 10.0
    # Battery LFP parameters
    bess_energy_kwh: float = 10.0
    bess_power_kw: float = 5.0
    soc_min: float = 0.10
    soc_max: float = 0.95
    soc_init: float = 0.50
    eta_ch: float = 0.955
    eta_dis: float = 0.955
    bess_standby_kw: float = 0.035
    cycle_deg_cost_pln_mwh: float = 160.0
    # Air-to-water heat pump; steady-state heat demand from one envelope UA value (no RC building model)
    hp_max_el_kw: float = 3.5
    ua_env_kw_per_k: float = 0.165    # envelope heat loss coefficient (165 W/K -> 5.8 kWth at -15°C)
    t_in_target_c: float = 21.3
    # EV charger (11 kW 3-phase Wallbox)
    ev_max_kw: float = 11.0
    ev_eta: float = 0.93
    # Grid connection limit
    grid_max_kw: float = 14.0


def simulate_rooftop_pv_kw(
    ts_utc: pd.DatetimeIndex,
    ghi_wm2: np.ndarray,
    dhi_wm2: np.ndarray,
    temp_air_c: np.ndarray,
    wind_ms: np.ndarray,
    pv_kwp: float = 10.0,
    inverter_kw: float = 10.0,
    tilt_deg: float = 35.0,
    azimuth_deg: float = 180.0,
    lat_deg: float = 52.23,
    lon_deg: float = 21.01,
) -> np.ndarray:
    """Compute 15-minute AC power (kW) of a crystalline-silicon rooftop PV array.

    Uses solar position, beam on the tilted plane plus isotropic sky diffuse and ground reflection,
    a Faiman-type module temperature model, and temperature coefficient gamma = -0.36 %/K.
    """
    # Day of year and fractional UTC hour
    doy = np.asarray(ts_utc.day_of_year, dtype=float)
    utc_hour = np.asarray(ts_utc.hour + ts_utc.minute / 60.0 + 0.125, dtype=float)

    # Solar declination and equation of time (Spencer / Cooper approximation)
    b = 2.0 * np.pi * (doy - 81.0) / 365.0
    eot_min = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
    decl_rad = np.deg2rad(23.45 * np.sin(2.0 * np.pi * (284.0 + doy) / 365.0))

    solar_time_h = utc_hour + lon_deg / 15.0 + eot_min / 60.0
    hra_rad = np.deg2rad(15.0 * (solar_time_h - 12.0))
    lat_rad = np.deg2rad(lat_deg)

    sin_el = np.sin(lat_rad) * np.sin(decl_rad) + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hra_rad)
    sin_el_pos = np.clip(sin_el, 0.0, 1.0)
    zen_rad = np.arccos(np.clip(sin_el, -1.0, 1.0))

    # Solar azimuth (0=N, 90=E, 180=S, 270=W)
    cos_az = (np.sin(decl_rad) - np.sin(lat_rad) * sin_el) / (np.cos(lat_rad) * np.sin(zen_rad) + 1e-9)
    az_rad = np.arccos(np.clip(cos_az, -1.0, 1.0))
    az_solar_deg = np.where(hra_rad > 0, 360.0 - np.rad2deg(az_rad), np.rad2deg(az_rad))

    # Angle of incidence on tilted plane
    tilt_rad = np.deg2rad(tilt_deg)
    d_az_rad = np.deg2rad(az_solar_deg - azimuth_deg)
    cos_aoi = sin_el_pos * np.cos(tilt_rad) + np.sin(zen_rad) * np.sin(tilt_rad) * np.cos(d_az_rad)
    cos_aoi = np.where(sin_el > 0.03, np.clip(cos_aoi, 0.0, 1.0), 0.0)

    dni_est = np.where(sin_el_pos > 0.06, np.maximum(0.0, (ghi_wm2 - dhi_wm2)) / np.maximum(0.06, sin_el_pos), 0.0)
    dni_est = np.clip(dni_est, 0.0, 1050.0)

    poa_beam = dni_est * cos_aoi
    poa_sky_diff = dhi_wm2 * (1.0 + np.cos(tilt_rad)) * 0.5
    poa_ground = ghi_wm2 * 0.20 * (1.0 - np.cos(tilt_rad)) * 0.5
    poa_wm2 = np.maximum(0.0, poa_beam + poa_sky_diff + poa_ground)

    # Faiman cell temperature model (u0 = 25.0, u1 = 6.84 W/m2K)
    t_cell = temp_air_c + poa_wm2 / (25.0 + 6.84 * np.maximum(0.5, wind_ms))
    temp_factor = 1.0 - 0.0036 * (t_cell - 25.0)
    dc_kw = pv_kwp * (poa_wm2 / 1000.0) * np.clip(temp_factor, 0.78, 1.12) * 0.96
    ac_kw = np.minimum(inverter_kw, dc_kw * 0.975)
    return np.where(ac_kw < 0.01, 0.0, ac_kw)


def heat_pump_cop(temp_out_c: np.ndarray, t_supply_c: float = 34.0) -> np.ndarray:
    """Realistic air-to-water heat pump COP curve (underfloor heating supply ~34°C, defrost dip near 0°C)."""
    lift = np.maximum(12.0, (t_supply_c + 4.0) - (temp_out_c - 6.0))
    cop_carnot = (273.15 + t_supply_c + 4.0) / lift
    # Defrost penalty around -3 to +3 °C
    defrost_pen = 0.08 * np.exp(-0.25 * (temp_out_c - 0.5) ** 2)
    cop = 0.44 * cop_carnot - defrost_pen
    return np.clip(cop, 1.85, 5.10)


def generate_household_profiles(
    ts_utc: pd.DatetimeIndex,
    temp_air_c: np.ndarray,
    ghi_wm2: np.ndarray,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic 15-minute household: appliance load, heating gains/season and EV plug-in schedule.

    None of these profiles is measured; they are parametric shapes with random noise (fixed seed).
    """
    rng = np.random.default_rng(seed)
    ts_local = ts_utc.tz_convert("Europe/Warsaw")
    n = len(ts_utc)
    hour = np.asarray(ts_local.hour + ts_local.minute / 60.0)
    weekday = np.asarray(ts_local.day_of_week)
    doy = np.asarray(ts_local.day_of_year)

    # 1. Base household electrical appliances (kW): ~5,000 kWh/yr
    morning_bump = 0.45 * np.exp(-0.5 * ((hour - 7.5) / 1.1) ** 2)
    evening_bump = 0.95 * np.exp(-0.5 * ((hour - 19.2) / 1.8) ** 2)
    midday_wknd = np.where(weekday >= 5, 0.40 * np.exp(-0.5 * ((hour - 13.5) / 2.0) ** 2), 0.12)
    seasonal = 1.0 + 0.14 * np.cos(2.0 * np.pi * (doy - 15.0) / 365.0)
    noise = rng.lognormal(mean=-0.03, sigma=0.22, size=n)
    base_load_kw = np.clip((0.24 + morning_bump + evening_bump + midday_wknd) * seasonal * noise, 0.14, 4.2)

    # 2. Space heating + DHW thermal demand
    # Heating season active when 24h rolling outdoor temp < 14.5 °C
    s_temp = pd.Series(temp_air_c).rolling(96, min_periods=1).mean().to_numpy()
    heating_active = s_temp < 14.5
    solar_gain_kwth = 0.0022 * ghi_wm2  # passive window solar gains (kWth)
    internal_gain_kwth = 0.35 + 0.30 * ((hour >= 17.0) | (hour < 7.0)).astype(float)

    # 3. EV schedule:
    # On weekdays plugged in 17:30 -> 07:00 next morning; on weekends plugged in 10:00 -> 16:00 and overnight
    # Daily trip energy requirement assigned at arrival (17:30 local time, or 10:00 Saturday/Sunday)
    ev_plugged = ((weekday < 5) & ((hour >= 17.5) | (hour < 7.0))) | (
        (weekday >= 5) & ((hour >= 10.0) | (hour < 8.0))
    )
    # Daily EV demand (kWh required per day): ~10.5 kWh/day (~3,800 kWh/yr = ~21,000 km/yr)
    is_arrival = (ts_local.hour == 17) & (ts_local.minute == 30)
    ev_req_kwh = np.where(is_arrival, rng.uniform(7.5, 13.5, size=n), 0.0)

    return pd.DataFrame(
        {
            "base_load_kw": base_load_kw,
            "heating_active": heating_active,
            "solar_gain_kwth": solar_gain_kwth,
            "internal_gain_kwth": internal_gain_kwth,
            "ev_plugged": ev_plugged,
            "ev_arrival_req_kwh": ev_req_kwh,
        },
        index=ts_utc,
    )
