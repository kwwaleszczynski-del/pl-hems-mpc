"""Home Assistant entity state coordinator for `pl_hems_mpc`.

Can be imported directly by Home Assistant Core (`async_setup_entry`) or executed standalone
in Hardware-in-the-Loop replay mode to generate live Home Assistant entity state snapshots.
"""

from __future__ import annotations

from typing import Any
from hems_mpc.hardware_profiles import compute_sg_ready_state, compute_ocpp_ev_current_a


def build_live_ha_entity_states(
    ts_local_iso: str,
    rce_pln_mwh: float,
    sdac_pln_mwh: float,
    total_buy_pln_mwh: float,
    pv_avail_kw: float,
    base_load_kw: float,
    bess_cmd_kw: float,
    soc_pct: float,
    ev_cmd_kw: float,
    th_buf_kwh: float,
    fifo_deposit_balance_pln: float,
    fifo_at_risk_80pct_pln: float,
    ci_delta_pct: float,
    inverter_brand: str = "deye_sg04lp3",
) -> dict[str, dict[str, Any]]:
    """Compute the 8 live Home Assistant entity states & attributes for a 15-minute step."""
    zero_exp = bool(rce_pln_mwh <= 0.0)
    exp_lim_w = 0 if zero_exp else 10_000
    bess_target_w = int(round(bess_cmd_kw * 1000.0))
    pv_surplus_kw = max(0.0, pv_avail_kw - base_load_kw)

    sg = compute_sg_ready_state(
        rce_pln_mwh=rce_pln_mwh,
        total_buy_pln_mwh=total_buy_pln_mwh,
        pv_surplus_kw=pv_surplus_kw,
        th_buffer_kwh=th_buf_kwh,
    )
    ev_amps = compute_ocpp_ev_current_a(ev_cmd_kw)

    if ci_delta_pct < 5.0:
        k_band, k_mult = "K1 (83% Discount)", 0.17
    elif ci_delta_pct < 10.0:
        k_band, k_mult = "K2 (50% Discount)", 0.50
    elif ci_delta_pct < 15.0:
        k_band, k_mult = "K3 (17% Discount)", 0.83
    else:
        k_band, k_mult = "K4 (No Discount)", 1.00

    return {
        "sensor.pl_hems_pse_rce_15min": {
            "state": round(float(rce_pln_mwh), 2),
            "unit_of_measurement": "PLN/MWh",
            "friendly_name": "PSE 15-min RCE Export Price",
            "attributes": {
                "timestamp_local": ts_local_iso,
                "zero_export_guard_active": zero_exp,
                "net_billing_credit_pln_mwh": round(max(0.0, float(rce_pln_mwh)), 2),
            },
        },
        "sensor.pl_hems_dynamic_import_price": {
            "state": round(float(total_buy_pln_mwh), 2),
            "unit_of_measurement": "PLN/MWh",
            "friendly_name": "Total Grid Import Price (SDAC + G12w Dist.)",
            "attributes": {"sdac_spot_pln_mwh": round(float(sdac_pln_mwh), 2)},
        },
        "sensor.pl_hems_fifo_deposit_balance": {
            "state": round(float(fifo_deposit_balance_pln), 2),
            "unit_of_measurement": "PLN",
            "friendly_name": "Prosumer 12M FIFO Deposit Balance",
            "attributes": {
                "refundable_cash_20pct_cap_pln": round(0.20 * float(fifo_deposit_balance_pln), 2),
                "at_risk_80pct_forfeiture_pln": round(float(fifo_at_risk_80pct_pln), 2),
            },
        },
        "sensor.pl_hems_ci_k_factor_live": {
            "state": k_band,
            "unit_of_measurement": "",
            "friendly_name": "PSE Capacity Fee Live Band (Art. 70a)",
            "attributes": {
                "peak_vs_offpeak_delta_pct": round(float(ci_delta_pct), 2),
                "statutory_multiplier_A": k_mult,
                "target_delta_for_k1_pct": 5.0,
            },
        },
        "sensor.pl_hems_mpc_bess_setpoint_w": {
            "state": bess_target_w,
            "unit_of_measurement": "W",
            "friendly_name": "MPC Battery Power Command (+Ch / -Dis)",
            "attributes": {
                "soc_pct": round(float(soc_pct), 1),
                "inverter_profile": inverter_brand,
                "export_limit_w": exp_lim_w,
            },
        },
        "binary_sensor.pl_hems_zero_export_guard": {
            "state": "ON" if zero_exp else "OFF",
            "unit_of_measurement": "",
            "friendly_name": "Negative-RCE Zero-Export Limit (Reg 40003=0W)",
            "attributes": {"export_limit_w": exp_lim_w},
        },
        "sensor.pl_hems_sg_ready_heat_pump": {
            "state": sg["state_code"],
            "unit_of_measurement": "",
            "friendly_name": "Heat Pump SG Ready Relay State",
            "attributes": {
                "sg_ready_mode": sg["mode_num"],
                "description": sg["label"],
                "slab_thermal_buffer_kwh": round(float(th_buf_kwh), 2),
            },
        },
        "sensor.pl_hems_ocpp_ev_current_a": {
            "state": ev_amps,
            "unit_of_measurement": "A",
            "friendly_name": "OCPP 1.6J Wallbox Pilot Current (3-Phase)",
            "attributes": {
                "charging_power_kw": round(ev_amps * 3 * 230.0 / 1000.0, 2) if ev_amps > 0 else 0.0,
            },
        },
    }
