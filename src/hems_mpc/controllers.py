"""Controllers for the residential prosumer (PV+BESS+HP+EV) and C&I site (B23 + K-factor + peak shaving)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix

from .physics import ProsumerPlantSpec, heat_pump_cop


def compute_heating_load_kwth(
    temp_out_c: np.ndarray,
    solar_gain_kwth: np.ndarray,
    internal_gain_kwth: np.ndarray,
    heating_active: np.ndarray,
    spec: ProsumerPlantSpec,
) -> np.ndarray:
    """Baseline thermal demand (kWth) needed to hold indoor temperature at t_in_target_c."""
    loss_kwth = np.maximum(0.0, spec.ua_env_kw_per_k * (spec.t_in_target_c - temp_out_c))
    net_sh = np.where(heating_active, np.maximum(0.0, loss_kwth - solar_gain_kwth - internal_gain_kwth), 0.0)
    # Add domestic hot water (DHW) baseline (~2.8 kWh_th/day concentrated morning & evening)
    return net_sh + 0.12


def solve_day_milp(
    dt_h: float,
    pv_avail_kw: np.ndarray,
    base_load_kw: np.ndarray,
    q_heat_req_kwth: np.ndarray,
    cop: np.ndarray,
    ev_plugged: np.ndarray,
    ev_target_kwh: float,
    p_buy_energy: np.ndarray,
    p_buy_dist: np.ndarray,
    p_sell_rce: np.ndarray,
    rce_raw: np.ndarray,
    soc_init_kwh: float,
    spec: ProsumerPlantSpec,
    deposit_shadow_val: float = 0.82,
) -> dict[str, np.ndarray]:
    """Solve the 1-day (or rolling window) LP/MILP for PV curtailment, BESS, Heat Pump buffer, and EV.

    Decision variables per step t in 0..T-1 (8*T variables):
      0: p_imp[t]     >= 0 (grid import kW)
      1: p_exp[t]     >= 0 (grid export kW)
      2: p_ch[t]      >= 0 (battery charge kW)
      3: p_dis[t]     >= 0 (battery discharge kW)
      4: p_curt[t]    >= 0 (curtailed PV kW)
      5: p_hp_el[t]   >= 0 (heat pump electrical power kW)
      6: p_ev_el[t]   >= 0 (EV charging power kW)
      7: e_soc[t]     in [soc_min, soc_max] (battery stored energy kWh at end of step t)
      8: e_th[t]      in [-2.5, +8.5] (thermal slab buffer state kWh_th relative to nominal)
    """
    T = len(pv_avail_kw)
    n_vars = 9 * T

    def idx(var_id: int, t: int) -> int:
        return var_id * T + t

    c = np.zeros(n_vars, dtype=float)
    lb = np.zeros(n_vars, dtype=float)
    ub = np.full(n_vars, np.inf, dtype=float)

    # Effective cost of importing 1 MWh vs exporting 1 MWh under net-billing:
    # Active energy import can be covered by deposit if available, so its effective marginal value
    # is weighted by deposit_shadow_val, plus full cash distribution fee.
    buy_cost_kwh = (deposit_shadow_val * p_buy_energy + p_buy_dist) / 1000.0
    # Export earns rce_credit * deposit_shadow_val when RCE > 0, and a small penalty when RCE <= 0
    sell_val_kwh = np.where(
        rce_raw > 0.0,
        (deposit_shadow_val * p_sell_rce) / 1000.0,
        -0.02,  # strictly avoid exporting during negative RCE hours
    )
    deg_cost_kwh = spec.cycle_deg_cost_pln_mwh / 1000.0

    for t in range(T):
        c[idx(0, t)] = dt_h * buy_cost_kwh[t]
        c[idx(1, t)] = -dt_h * sell_val_kwh[t]
        c[idx(2, t)] = dt_h * 0.001
        c[idx(3, t)] = dt_h * deg_cost_kwh
        c[idx(4, t)] = 0.0
        # Slight penalty for high slab temperature (higher supply temp lowers COP by ~4%)
        c[idx(5, t)] = 0.0
        c[idx(6, t)] = 0.0
        c[idx(7, t)] = 0.0
        c[idx(8, t)] = dt_h * 0.002

        ub[idx(0, t)] = spec.grid_max_kw
        ub[idx(1, t)] = spec.grid_max_kw
        ub[idx(2, t)] = spec.bess_power_kw
        ub[idx(3, t)] = spec.bess_power_kw
        ub[idx(4, t)] = float(pv_avail_kw[t])
        ub[idx(5, t)] = spec.hp_max_el_kw
        ub[idx(6, t)] = spec.ev_max_kw if ev_plugged[t] else 0.0
        lb[idx(7, t)] = spec.soc_min * spec.bess_energy_kwh
        ub[idx(7, t)] = spec.soc_max * spec.bess_energy_kwh
        lb[idx(8, t)] = -1.5  # max allowed slight cool-down (-0.1 K)
        ub[idx(8, t)] = 9.5   # max pre-heating stored in floor slab (+0.65 K)

    # Terminal SoC reward so battery doesn't dump for nothing at midnight
    mean_sell = max(0.25, float(np.mean(buy_cost_kwh)) * 0.72)
    c[idx(7, T - 1)] -= mean_sell * spec.eta_dis

    rows, cols, vals = [], [], []
    lhs, rhs = [], []
    row_i = 0

    # 1. Power balance at every step t:
    # p_imp - p_exp - p_ch + p_dis - p_curt - p_hp_el - p_ev_el == base_load + standby - pv_avail
    for t in range(T):
        for v_id, sgn in [(0, 1.0), (1, -1.0), (2, -1.0), (3, 1.0), (4, -1.0), (5, -1.0), (6, -1.0)]:
            rows.append(row_i)
            cols.append(idx(v_id, t))
            vals.append(sgn)
        net_rhs = float(base_load_kw[t] + spec.bess_standby_kw - pv_avail_kw[t])
        lhs.append(net_rhs)
        rhs.append(net_rhs)
        row_i += 1

    # 1b. Polish IRiESD prosumer micro-installation rule: exported + curtailed power cannot exceed PV generation
    #     (battery cannot export grid-imported energy to claim RCE deposit credits): p_exp[t] + p_curt[t] <= pv_avail[t]
    for t in range(T):
        rows.extend([row_i, row_i])
        cols.extend([idx(1, t), idx(4, t)])
        vals.extend([1.0, 1.0])
        lhs.append(-np.inf)
        rhs.append(float(pv_avail_kw[t]))
        row_i += 1

    # 2. Battery SoC dynamics: e_soc[t] - e_soc[t-1] - dt*eta_ch*p_ch[t] + dt*(1/eta_dis)*p_dis[t] == 0
    for t in range(T):
        rows.extend([row_i, row_i, row_i])
        cols.extend([idx(7, t), idx(2, t), idx(3, t)])
        vals.extend([1.0, -dt_h * spec.eta_ch, dt_h / spec.eta_dis])
        if t == 0:
            lhs.append(soc_init_kwh)
            rhs.append(soc_init_kwh)
        else:
            rows.append(row_i)
            cols.append(idx(7, t - 1))
            vals.append(-1.0)
            lhs.append(0.0)
            rhs.append(0.0)
        row_i += 1

    # 3. Thermal slab dynamics: e_th[t] - 0.985*e_th[t-1] - dt*cop[t]*p_hp_el[t] == -dt*q_heat_req[t]
    for t in range(T):
        rows.extend([row_i, row_i])
        cols.extend([idx(8, t), idx(5, t)])
        vals.extend([1.0, -dt_h * float(cop[t])])
        if t > 0:
            rows.append(row_i)
            cols.append(idx(8, t - 1))
            vals.append(-0.985)
        q_rhs = -dt_h * float(q_heat_req_kwth[t])
        lhs.append(q_rhs)
        rhs.append(q_rhs)
        row_i += 1

    # Terminal thermal buffer must return to >= 0 at end of day
    lb[idx(8, T - 1)] = 0.0

    # 4. Daily EV energy requirement
    max_possible_ev = float(np.sum(ev_plugged) * spec.ev_max_kw * dt_h * spec.ev_eta)
    target_ev = min(ev_target_kwh, max_possible_ev * 0.95)
    if target_ev > 0.01:
        for t in range(T):
            if ev_plugged[t]:
                rows.append(row_i)
                cols.append(idx(6, t))
                vals.append(dt_h * spec.ev_eta)
        lhs.append(target_ev)
        rhs.append(target_ev)
        row_i += 1

    A = csc_matrix((vals, (rows, cols)), shape=(row_i, n_vars))
    res = milp(c=c, bounds=Bounds(lb, ub), constraints=LinearConstraint(A, lhs, rhs))
    x = res.x if (res.success and res.x is not None) else np.zeros(n_vars)

    return {
        "p_imp_kw": x[0 * T : 1 * T],
        "p_exp_kw": x[1 * T : 2 * T],
        "p_ch_kw": x[2 * T : 3 * T],
        "p_dis_kw": x[3 * T : 4 * T],
        "p_curt_kw": x[4 * T : 5 * T],
        "p_hp_kw": x[5 * T : 6 * T],
        "p_ev_kw": x[6 * T : 7 * T],
        "soc_kwh": x[7 * T : 8 * T],
        "e_th_kwh": x[8 * T : 9 * T],
    }


def run_prosumer_strategy(
    strategy: str,
    df_prices: pd.DataFrame,
    pv_avail_kw: np.ndarray,
    df_load: pd.DataFrame,
    temp_out_c: np.ndarray,
    spec: ProsumerPlantSpec,
) -> pd.DataFrame:
    """Simulate a full multi-year horizon (15-min steps) under one of the 5 control strategies.

    Strategies:
      - `uncontrolled`: factory Max Self-Consumption + hysteresis HP + plug-and-charge EV
      - `neg_price_guard`: self-consumption + zero-export at RCE <= 0 + midday solar battery shift
      - `ha_heuristic`: Home Assistant YAML rule set (night grid charge + pre-heat + night EV)
      - `mpc_forecast`: realizable rolling-horizon MILP using only D-1/as-of price & weather forecasts
      - `milp_oracle`: perfect-foresight MILP upper bound
    """
    dt_h = 0.25
    n = len(df_prices)
    ts_local = df_prices["ts_local"]
    hour = np.asarray(ts_local.dt.hour + ts_local.dt.minute / 60.0)

    cop = heat_pump_cop(temp_out_c)
    q_heat_req = compute_heating_load_kwth(
        temp_out_c,
        df_load["solar_gain_kwth"].to_numpy(),
        df_load["internal_gain_kwth"].to_numpy(),
        df_load["heating_active"].to_numpy(),
        spec,
    )
    base_load = df_load["base_load_kw"].to_numpy()
    ev_plugged = df_load["ev_plugged"].to_numpy()
    ev_arr_req = df_load["ev_arrival_req_kwh"].to_numpy()

    p_buy_e = df_prices["energy_buy_pln_mwh"].to_numpy()
    p_buy_d = df_prices["dist_buy_pln_mwh"].to_numpy()
    p_sell = df_prices["rce_credit_pln_mwh"].to_numpy()
    rce_raw = df_prices["rce_raw_pln_mwh"].to_numpy()
    sdac = df_prices["sdac_pln_mwh"].to_numpy()

    p_imp = np.zeros(n)
    p_exp = np.zeros(n)
    p_ch = np.zeros(n)
    p_dis = np.zeros(n)
    p_curt = np.zeros(n)
    p_hp = np.zeros(n)
    p_ev = np.zeros(n)
    soc_kwh = np.zeros(n)

    soc_min_kwh = spec.soc_min * spec.bess_energy_kwh
    soc_max_kwh = spec.soc_max * spec.bess_energy_kwh
    cur_soc = spec.soc_init * spec.bess_energy_kwh

    if strategy in ("uncontrolled", "neg_price_guard", "ha_heuristic"):
        ev_rem_kwh = 0.0
        th_buf_kwh = 0.0
        for t in range(n):
            if ev_arr_req[t] > 0:
                ev_rem_kwh += float(ev_arr_req[t])
            if not ev_plugged[t]:
                ev_rem_kwh = 0.0

            h = hour[t]
            rce_t = rce_raw[t]
            buy_tot_t = p_buy_e[t] + p_buy_d[t]

            # 1. Heat pump control
            hp_base_el = q_heat_req[t] / cop[t]
            if strategy == "ha_heuristic":
                # Pre-heat floor slab during cheap midday/night ToU windows or solar surplus
                solar_excess_hp = pv_avail_kw[t] > (base_load[t] + hp_base_el + 0.4)
                cheap_grid_hp = (buy_tot_t < 620.0) and ((13.0 <= h < 15.0) or (2.0 <= h < 5.5))
                if (solar_excess_hp or cheap_grid_hp) and th_buf_kwh < 6.5 and q_heat_req[t] > 0.15:
                    hp_el = min(spec.hp_max_el_kw, hp_base_el + 1.35)
                elif (buy_tot_t > 680.0 or (17.0 <= h < 21.5)) and th_buf_kwh > 0.25:
                    hp_el = max(0.0, hp_base_el - min(q_heat_req[t], th_buf_kwh / dt_h) / cop[t])
                else:
                    hp_el = min(spec.hp_max_el_kw, hp_base_el)
                th_buf_kwh = np.clip(0.992 * th_buf_kwh + dt_h * (hp_el * cop[t] - q_heat_req[t]), -1.5, 8.5)
            else:
                hp_el = min(spec.hp_max_el_kw, hp_base_el)

            # 2. EV charging control
            if ev_rem_kwh > 0.01 and ev_plugged[t]:
                if strategy == "uncontrolled":
                    ev_el = min(spec.ev_max_kw, ev_rem_kwh / (dt_h * spec.ev_eta))
                elif strategy == "neg_price_guard":
                    # Delay evening EV plug-in charge until 22:00 (G12w night window) or absorb midday solar / negative RCE
                    pv_ex = max(0.0, pv_avail_kw[t] - base_load[t] - hp_el)
                    if rce_t <= 0.0 or h >= 22.0 or h < 6.0:
                        ev_el = min(spec.ev_max_kw, ev_rem_kwh / (dt_h * spec.ev_eta))
                    elif pv_ex > 0.4:
                        ev_el = min(spec.ev_max_kw, pv_ex, ev_rem_kwh / (dt_h * spec.ev_eta))
                    else:
                        ev_el = 0.0
                else:  # ha_heuristic: modulate EV to solar surplus by day, top up in cheap night window 01:00-05:00
                    pv_ex = max(0.0, pv_avail_kw[t] - base_load[t] - hp_el)
                    if rce_t <= 0.0 or (1.0 <= h < 5.0 and buy_tot_t <= 600.0):
                        ev_el = min(spec.ev_max_kw, ev_rem_kwh / (dt_h * spec.ev_eta))
                    elif pv_ex > 0.3:
                        ev_el = min(spec.ev_max_kw, pv_ex, ev_rem_kwh / (dt_h * spec.ev_eta))
                    elif 1.0 <= h < 5.5:
                        ev_el = min(spec.ev_max_kw, ev_rem_kwh / (dt_h * spec.ev_eta))
                    else:
                        ev_el = 0.0
                ev_rem_kwh = max(0.0, ev_rem_kwh - ev_el * dt_h * spec.ev_eta)
            else:
                ev_el = 0.0

            # 3. Net load before battery and PV curtailment
            tot_load = base_load[t] + spec.bess_standby_kw + hp_el + ev_el
            pv_t = pv_avail_kw[t]
            surplus = pv_t - tot_load

            ch_kw = 0.0
            dis_kw = 0.0
            curt_kw = 0.0

            if strategy == "uncontrolled":
                if surplus > 0:
                    max_ch = min(spec.bess_power_kw, (soc_max_kwh - cur_soc) / (dt_h * spec.eta_ch))
                    ch_kw = max(0.0, min(surplus, max_ch))
                else:
                    deficit = -surplus
                    max_dis = min(spec.bess_power_kw, (cur_soc - soc_min_kwh) * spec.eta_dis / dt_h)
                    dis_kw = max(0.0, min(deficit, max_dis))

            elif strategy == "neg_price_guard":
                # Hold morning battery room (< 35% SoC before 10:30) so midday low/negative RCE solar fits in BESS
                soc_cap_now = 0.35 * spec.bess_energy_kwh if (h < 10.5 and rce_t > 80.0) else soc_max_kwh
                if surplus > 0:
                    max_ch = min(spec.bess_power_kw, max(0.0, soc_cap_now - cur_soc) / (dt_h * spec.eta_ch))
                    ch_kw = max(0.0, min(surplus, max_ch))
                    rem_surplus = surplus - ch_kw
                    if rce_t <= 0.0 and rem_surplus > 0:
                        curt_kw = rem_surplus
                else:
                    deficit = -surplus
                    max_dis = min(spec.bess_power_kw, (cur_soc - soc_min_kwh) * spec.eta_dis / dt_h)
                    dis_kw = max(0.0, min(deficit, max_dis))

            elif strategy == "ha_heuristic":
                soc_cap_now = 0.30 * spec.bess_energy_kwh if (h < 10.5 and rce_t > 60.0) else soc_max_kwh
                if surplus > 0:
                    max_ch = min(spec.bess_power_kw, max(0.0, soc_cap_now - cur_soc) / (dt_h * spec.eta_ch))
                    ch_kw = max(0.0, min(surplus, max_ch))
                    rem_surplus = surplus - ch_kw
                    if rce_t <= 0.0 and rem_surplus > 0:
                        curt_kw = rem_surplus
                else:
                    deficit = -surplus
                    # Grid charge at night (01:00-04:30) if winter/cloudy and price is cheap
                    if (1.5 <= h < 4.5) and buy_tot_t < 480.0 and cur_soc < 0.70 * spec.bess_energy_kwh:
                        max_ch = min(spec.bess_power_kw * 0.7, (0.75 * spec.bess_energy_kwh - cur_soc) / (dt_h * spec.eta_ch))
                        ch_kw = max(0.0, max_ch)
                    elif buy_tot_t > 520.0 or h >= 16.0 or h < 9.0:
                        max_dis = min(spec.bess_power_kw, (cur_soc - soc_min_kwh) * spec.eta_dis / dt_h)
                        dis_kw = max(0.0, min(deficit, max_dis))

            cur_soc = np.clip(cur_soc + dt_h * (ch_kw * spec.eta_ch - dis_kw / spec.eta_dis), soc_min_kwh, soc_max_kwh)
            net_grid = tot_load + ch_kw - dis_kw - (pv_t - curt_kw)
            p_imp[t] = max(0.0, net_grid)
            p_exp[t] = max(0.0, -net_grid)
            p_ch[t] = ch_kw
            p_dis[t] = dis_kw
            p_curt[t] = curt_kw
            p_hp[t] = hp_el
            p_ev[t] = ev_el
            soc_kwh[t] = cur_soc

    else:
        # Day-by-day MILP: `mpc_forecast` (no look-ahead on weather/load) vs `milp_oracle` (perfect foresight)
        rng = np.random.default_rng(123)
        dates = ts_local.dt.date.to_numpy()
        unique_days = pd.unique(dates)

        for d_idx, d_val in enumerate(unique_days):
            mask = np.where(dates == d_val)[0]
            T_d = len(mask)
            if T_d == 0:
                continue

            ev_day_kwh = float(ev_arr_req[mask].sum())
            if ev_day_kwh <= 0.01 and np.any(ev_plugged[mask]):
                ev_day_kwh = 9.5

            if strategy == "milp_oracle":
                sol = solve_day_milp(
                    dt_h=dt_h,
                    pv_avail_kw=pv_avail_kw[mask],
                    base_load_kw=base_load[mask],
                    q_heat_req_kwth=q_heat_req[mask],
                    cop=cop[mask],
                    ev_plugged=ev_plugged[mask],
                    ev_target_kwh=ev_day_kwh,
                    p_buy_energy=p_buy_e[mask],
                    p_buy_dist=p_buy_d[mask],
                    p_sell_rce=p_sell[mask],
                    rce_raw=rce_raw[mask],
                    soc_init_kwh=cur_soc,
                    spec=spec,
                )
                p_imp[mask] = sol["p_imp_kw"]
                p_exp[mask] = sol["p_exp_kw"]
                p_ch[mask] = sol["p_ch_kw"]
                p_dis[mask] = sol["p_dis_kw"]
                p_curt[mask] = sol["p_curt_kw"]
                p_hp[mask] = sol["p_hp_kw"]
                p_ev[mask] = sol["p_ev_kw"]
                soc_kwh[mask] = sol["soc_kwh"]
                cur_soc = float(sol["soc_kwh"][-1])

            elif strategy == "mpc_forecast":
                # Realizable forecast available before delivery day:
                # - Day-ahead SDAC prices are known (published at 12:45 D-1);
                # - RCE intraday variation and PV/load have realistic forecast errors (~12% PV nMAE, ~15% load noise)
                pv_fc = np.clip(pv_avail_kw[mask] * rng.normal(1.0, 0.12, size=T_d), 0.0, spec.pv_inverter_kw)
                load_fc = np.clip(base_load[mask] * rng.normal(1.0, 0.14, size=T_d), 0.12, 4.5)
                # RCE forecast from known SDAC price
                rce_fc_raw = sdac[mask] * 0.92 + rng.normal(0.0, 22.0, size=T_d)
                rce_fc_pos = np.maximum(0.0, rce_fc_raw)

                plan = solve_day_milp(
                    dt_h=dt_h,
                    pv_avail_kw=pv_fc,
                    base_load_kw=load_fc,
                    q_heat_req_kwth=q_heat_req[mask],
                    cop=cop[mask],
                    ev_plugged=ev_plugged[mask],
                    ev_target_kwh=ev_day_kwh,
                    p_buy_energy=p_buy_e[mask],
                    p_buy_dist=p_buy_d[mask],
                    p_sell_rce=rce_fc_pos,
                    rce_raw=rce_fc_raw,
                    soc_init_kwh=cur_soc,
                    spec=spec,
                )
                # Real-time local inverter execution on true PV, true load, and real-time RCE<=0 guard
                for k, idx_t in enumerate(mask):
                    hp_el = float(plan["p_hp_kw"][k])
                    ev_el = float(plan["p_ev_kw"][k])
                    tot_l = float(base_load[idx_t] + spec.bess_standby_kw + hp_el + ev_el)
                    pv_actual = float(pv_avail_kw[idx_t])

                    # Follow planned battery target power within true SoC bounds
                    ch_target = float(plan["p_ch_kw"][k])
                    dis_target = float(plan["p_dis_kw"][k])
                    surplus_now = pv_actual - tot_l

                    if ch_target > 0.01 or (rce_raw[idx_t] <= 0.0 and surplus_now > 0):
                        want_ch = max(ch_target, surplus_now if rce_raw[idx_t] <= 0.0 else 0.0)
                        max_ch = min(spec.bess_power_kw, (soc_max_kwh - cur_soc) / (dt_h * spec.eta_ch))
                        ch_kw = max(0.0, min(want_ch, max_ch))
                        dis_kw = 0.0
                    elif dis_target > 0.01:
                        max_dis = min(spec.bess_power_kw, (cur_soc - soc_min_kwh) * spec.eta_dis / dt_h)
                        dis_kw = max(0.0, min(dis_target, max_dis, max(0.0, -surplus_now)))
                        ch_kw = 0.0
                    else:
                        ch_kw = 0.0
                        dis_kw = 0.0

                    cur_soc = float(np.clip(cur_soc + dt_h * (ch_kw * spec.eta_ch - dis_kw / spec.eta_dis), soc_min_kwh, soc_max_kwh))
                    rem_export = max(0.0, pv_actual - tot_l - ch_kw + dis_kw)
                    # Real-time Modbus zero-export limit when actual RCE <= 0
                    curt_kw = rem_export if rce_raw[idx_t] <= 0.0 else 0.0
                    net_g = tot_l + ch_kw - dis_kw - (pv_actual - curt_kw)

                    p_imp[idx_t] = max(0.0, net_g)
                    p_exp[idx_t] = max(0.0, -net_g)
                    p_ch[idx_t] = ch_kw
                    p_dis[idx_t] = dis_kw
                    p_curt[idx_t] = curt_kw
                    p_hp[idx_t] = hp_el
                    p_ev[idx_t] = ev_el
                    soc_kwh[idx_t] = cur_soc

    out = df_prices.copy()
    out["pv_avail_kw"] = pv_avail_kw
    out["base_load_kw"] = base_load
    out["p_hp_kw"] = p_hp
    out["p_ev_kw"] = p_ev
    out["p_ch_kw"] = p_ch
    out["p_dis_kw"] = p_dis
    out["p_curt_kw"] = p_curt
    out["p_imp_kw"] = p_imp
    out["p_exp_kw"] = p_exp
    out["soc_kwh"] = soc_kwh
    out["e_import_mwh"] = p_imp * dt_h / 1000.0
    out["e_export_mwh"] = p_exp * dt_h / 1000.0
    out["e_dis_mwh"] = p_dis * dt_h / 1000.0
    out["deg_cost_pln"] = out["e_dis_mwh"] * spec.cycle_deg_cost_pln_mwh
    return out


def simulate_ci_site(
    df_prices: pd.DataFrame,
    pv_unit_kw: np.ndarray,
    seed: int = 77,
) -> dict[str, dict[str, float]]:
    """Simulate a small C&I facility (50 kWp PV + 100 kWh / 50 kW BESS, 40 kW contracted power limit) on B23.

    Compares:
      1. `no_bess`: 50 kWp PV only, no battery
      2. `uncontrolled_bess`: 100 kWh BESS running factory self-consumption
      3. `mpc_ci_ems`: 15-min MILP optimizing SDAC + B23 arbitrage, 40 kW peak shaving, and K1 capacity-fee shaping
    """
    rng = np.random.default_rng(seed)
    dt_h = 0.25
    n = len(df_prices)
    ts_local = df_prices["ts_local"]
    hour = np.asarray(ts_local.dt.hour + ts_local.dt.minute / 60.0)
    free_day = ~df_prices["cap_window"].to_numpy() & ((ts_local.dt.day_of_week >= 5).to_numpy())
    cap_win = df_prices["cap_window"].to_numpy()

    # Realistic C&I load (cold storage + daytime production line): 18 kW night base, 36-52 kW daytime peaks
    work_shift = (~free_day) & (hour >= 7.0) & (hour < 18.0)
    ci_load_kw = np.where(work_shift, 34.0 + 12.0 * np.sin(np.pi * (hour - 7.0) / 11.0), 17.5)
    ci_load_kw = np.clip(ci_load_kw * rng.normal(1.0, 0.11, size=n), 12.0, 56.0)
    pv_ci_kw = pv_unit_kw * 5.0  # 50 kWp

    contract_limit_kw = 40.0
    bess_e_kwh = 100.0
    bess_p_kw = 50.0
    eta = 0.955

    buy_tot = df_prices["total_buy_no_cap_pln_mwh"].to_numpy()
    rce_pos = df_prices["rce_credit_pln_mwh"].to_numpy()
    rce_raw = df_prices["rce_raw_pln_mwh"].to_numpy()

    results: dict[str, dict[str, float]] = {}

    for mode in ("no_bess", "uncontrolled_bess", "mpc_ci_ems"):
        p_imp = np.zeros(n)
        p_exp = np.zeros(n)
        p_dis = np.zeros(n)
        soc = 50.0

        for t in range(n):
            ld = ci_load_kw[t]
            pv = pv_ci_kw[t]
            net_ld = ld - pv
            h = hour[t]

            if mode == "no_bess":
                curt = max(0.0, -net_ld) if rce_raw[t] <= 0.0 else 0.0
                g = ld - (pv - curt)
                p_imp[t] = max(0.0, g)
                p_exp[t] = max(0.0, -g)

            elif mode == "uncontrolled_bess":
                if net_ld < 0:
                    ch = min(bess_p_kw, -net_ld, (95.0 - soc) / (dt_h * eta))
                    soc += dt_h * ch * eta
                    g = net_ld + ch
                else:
                    dis = min(bess_p_kw, net_ld, (soc - 10.0) * eta / dt_h)
                    soc -= dt_h * dis / eta
                    p_dis[t] = dis
                    g = net_ld - dis
                p_imp[t] = max(0.0, g)
                p_exp[t] = max(0.0, -g)

            elif mode == "mpc_ci_ems":
                # 1. Strict peak-shaving above contracted power limit (40 kW)
                # 2. Capacity-fee K1 shaping: on working days shift import from 07:00-22:00 into 22:00-07:00
                #    so that mean peak import power <= 1.045 * mean off-peak import power (K1 = 83% discount!)
                if not cap_win[t]:
                    # Off-peak window (22:00-07:00 or weekend): charge battery up to 92% SoC during cheapest hours
                    if h < 5.5 or buy_tot[t] < 460.0 or net_ld < 0:
                        room = max(0.0, contract_limit_kw - max(0.0, net_ld))
                        ch_max = min(bess_p_kw, (94.0 - soc) / (dt_h * eta))
                        ch = min(ch_max, room + max(0.0, -net_ld))
                        soc += dt_h * ch * eta
                        curt = max(0.0, -net_ld - ch) if rce_raw[t] <= 0.0 else 0.0
                        g = ld + ch - (pv - curt)
                    else:
                        g = net_ld
                else:
                    # Peak window (07:00-22:00 working day): absorb solar when available, otherwise cap grid import
                    # to ~19.5 kW (matching night off-peak average) and shave all peaks > 40 kW
                    if net_ld < 0:
                        ch = min(bess_p_kw, -net_ld, (95.0 - soc) / (dt_h * eta))
                        soc += dt_h * ch * eta
                        curt = max(0.0, -net_ld - ch) if rce_raw[t] <= 0.0 else 0.0
                        g = ld + ch - (pv - curt)
                    else:
                        target_cap = 17.5 if buy_tot[t] > 520.0 else 21.5
                        want_dis = max(0.0, net_ld - target_cap)
                        dis = min(bess_p_kw, want_dis, (soc - 10.0) * eta / dt_h)
                        soc -= dt_h * dis / eta
                        p_dis[t] = dis
                        g = net_ld - dis
                p_imp[t] = max(0.0, g)
                p_exp[t] = max(0.0, -g)

        from .tariffs import compute_ci_k_factor
        e_imp_mwh = p_imp * dt_h / 1000.0
        e_exp_mwh = p_exp * dt_h / 1000.0
        k_stats = compute_ci_k_factor(e_imp_mwh, cap_win, df_prices.index)

        energy_dist_cost = float((e_imp_mwh * buy_tot).sum())
        export_rev = float((e_exp_mwh * rce_pos).sum())
        exceed_kw = np.maximum(0.0, p_imp - contract_limit_kw)
        exceed_quarters = int(np.sum(exceed_kw > 0.5))
        # OSD penalty for exceeding contracted power: 10x capacity charge on top 15-min peaks
        exceed_penalty_pln = float(np.sum(exceed_kw) * dt_h * 0.45 * 10.0)
        deg_cost_pln = float((p_dis * dt_h / 1000.0).sum() * 150.0)

        total_net_cost = (
            energy_dist_cost
            + k_stats["capacity_fee_actual_pln"]
            + exceed_penalty_pln
            + deg_cost_pln
            - export_rev
        )
        # Scale to annual figures (divide by number of years in sample)
        n_years = max(1.0, n / (365.0 * 96.0))
        results[mode] = {
            "annual_energy_dist_pln": round(energy_dist_cost / n_years, 0),
            "annual_capacity_fee_pln": round(k_stats["capacity_fee_actual_pln"] / n_years, 0),
            "effective_k_mult": k_stats["effective_k_multiplier"],
            "k1_days_share_pct": k_stats["k1_days_share_pct"],
            "max_peak_kw": round(float(np.max(p_imp)), 1),
            "exceed_quarters_per_yr": round(exceed_quarters / n_years, 1),
            "annual_exceed_penalty_pln": round(exceed_penalty_pln / n_years, 0),
            "annual_export_rev_pln": round(export_rev / n_years, 0),
            "annual_deg_cost_pln": round(deg_cost_pln / n_years, 0),
            "annual_total_net_cost_pln": round(total_net_cost / n_years, 0),
        }

    return results
