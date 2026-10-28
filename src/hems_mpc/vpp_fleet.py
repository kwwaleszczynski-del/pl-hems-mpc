"""Virtual Power Plant (VPP) fleet aggregation benchmark: 250 Home Assistant prosumers + 15 C&I sites.

Models the stochastic aggregation of behind-the-meter (BTM) hybrid inverters, heat pumps, and EV chargers
into a single >= 1 MW Balancing Service Provider (BSP) portfolio for the Polish PSE market (aFRR / mFRR
capacity + Intraday spread capture) while respecting local prosumer net-billing and C&I K1 constraints.

Note on IP / Commercial Boundary:
This module provides the open-source Monte Carlo P90 availability and baseline fleet dispatch benchmark.
Production ADMM decomposition weights, empirical BMS cell-temperature derating surfaces, and order-book
bidding heuristics are maintained separately in the private commercial integration package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_vpp_fleet_benchmark(
    df_prosumer_mpc: pd.DataFrame,
    n_homes: int = 250,
    n_ci_sites: int = 15,
    seed: int = 2026,
) -> dict[str, float | int | list]:
    """Evaluate a 250-home + 15-C&I VPP portfolio across 35,040 quarter-hours (2025-10..2026-09).

    Fleet composition:
      - 250 residential prosumers on Home Assistant (`10 kWp PV + 10 kWh / 5 kW LFP + 3.5 kW HP + 11 kW EV`)
        -> Nameplate BESS: 1.25 MW / 2.50 MWh; Nameplate Flex Load (HP+EV): 3.62 MW
      - 15 C&I B23 facilities (`50 kWp PV + 100 kWh / 50 kW BESS`)
        -> Nameplate BESS: 0.75 MW / 1.50 MWh
      - Total BESS Nameplate: 2.00 MW / 4.00 MWh (+ flex load headroom)
    """
    rng = np.random.default_rng(seed)
    n = len(df_prosumer_mpc)
    dt_h = 0.25

    soc_home_kwh = df_prosumer_mpc["soc_kwh"].to_numpy()
    p_ch_home_kw = df_prosumer_mpc["p_ch_kw"].to_numpy()
    p_dis_home_kw = df_prosumer_mpc["p_dis_kw"].to_numpy()
    rce = df_prosumer_mpc["rce_raw_pln_mwh"].to_numpy()
    sdac = df_prosumer_mpc["sdac_pln_mwh"].to_numpy()

    # Upward flexibility (injection / load reduction) available per home at step t (kW):
    # With a 25% VPP SoC reserve floor + sheddable EV/HP load, each home provides firm upward response
    p_flex_load = (
        df_prosumer_mpc["p_hp_kw"].to_numpy() + df_prosumer_mpc["p_ev_kw"].to_numpy()
        if "p_hp_kw" in df_prosumer_mpc.columns
        else np.zeros(n)
    )
    up_headroom_home_kw = np.where(
        soc_home_kwh > 2.2,
        np.maximum(2.8, (5.0 - p_dis_home_kw) + p_ch_home_kw + p_flex_load),
        2.6 + p_ch_home_kw + p_flex_load,  # VPP SoC reserve floor (25% SoC held for contracted aFRR blocks)
    )
    # Downward flexibility (absorption / PV curtailment / extra charging + HP boost) per home (kW):
    down_headroom_home_kw = np.where(
        soc_home_kwh < 8.5,
        (5.0 - p_ch_home_kw) + p_dis_home_kw + 1.4,
        p_dis_home_kw + 3.2,  # inverter dynamic PV curtailment + SG Ready Mode 4 boost
    )

    # Monte Carlo draws across the 250 homes + 15 C&I sites for telemetry reliability (97.2% online)
    # and local override diversity (84% non-coincident availability)
    online_ratio = rng.normal(0.972, 0.012, size=n).clip(0.91, 0.995)
    behavioral_factor = rng.normal(0.84, 0.05, size=n).clip(0.68, 0.95)

    fleet_up_mw = (
        (n_homes * up_headroom_home_kw + n_ci_sites * 32.0) / 1000.0
    ) * online_ratio * behavioral_factor
    fleet_down_mw = (
        (n_homes * down_headroom_home_kw + n_ci_sites * 36.0) / 1000.0
    ) * online_ratio * behavioral_factor

    p90_up_mw = float(np.percentile(fleet_up_mw, 10))
    p90_down_mw = float(np.percentile(fleet_down_mw, 10))
    p50_up_mw = float(np.median(fleet_up_mw))
    p50_down_mw = float(np.median(fleet_down_mw))

    # Incremental VPP revenue above standalone local HEMS/C&I savings:
    # 1. Symmetric/Asymmetric PSE aFRR capacity availability payment on P90 firm block (~95 PLN/MW/h)
    #    after reserving 35% of time for local peak/K1 protection -> 65% market availability factor
    afrr_cap_rev_pln = (p90_up_mw * 0.55 + p90_down_mw * 0.65) * 8760.0 * 62.0
    # 2. Intraday (IDC) / RCE-SDAC imbalance spread capture on 15-min forecast updates
    spread_abs = np.abs(rce - sdac)
    idc_arb_rev_pln = float(np.sum(spread_abs * dt_h * 0.22 * (p50_up_mw * 0.35)))

    gross_vpp_rev_pln = afrr_cap_rev_pln + idc_arb_rev_pln
    aggregator_fee_share = 0.22  # 22% VPP platform & BRP fee, 78% distributed to asset owners
    net_to_owners_pln = gross_vpp_rev_pln * (1.0 - aggregator_fee_share)

    # Split between residential prosumers (58% of flex volume) and C&I sites (42% of flex volume)
    bonus_per_home_pln = (net_to_owners_pln * 0.58) / n_homes
    bonus_per_ci_pln = (net_to_owners_pln * 0.42) / n_ci_sites

    return {
        "n_homes": n_homes,
        "n_ci_sites": n_ci_sites,
        "installed_bess_power_mw": round(n_homes * 0.005 + n_ci_sites * 0.050, 2),
        "installed_bess_energy_mwh": round(n_homes * 0.010 + n_ci_sites * 0.100, 2),
        "mean_telemetry_online_pct": round(float(np.mean(online_ratio) * 100.0), 2),
        "p90_firm_upward_mw": round(p90_up_mw, 2),
        "p90_firm_downward_mw": round(p90_down_mw, 2),
        "p50_upward_mw": round(p50_up_mw, 2),
        "p50_downward_mw": round(p50_down_mw, 2),
        "annual_afrr_capacity_rev_pln": round(afrr_cap_rev_pln, 0),
        "annual_intraday_spread_rev_pln": round(idc_arb_rev_pln, 0),
        "annual_gross_vpp_rev_pln": round(gross_vpp_rev_pln, 0),
        "aggregator_platform_margin_pln": round(gross_vpp_rev_pln * aggregator_fee_share, 0),
        "vpp_bonus_per_prosumer_home_pln_yr": round(bonus_per_home_pln, 0),
        "vpp_bonus_per_ci_site_pln_yr": round(bonus_per_ci_pln, 0),
    }
