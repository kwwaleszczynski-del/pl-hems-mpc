"""Property tests for tariffs, controllers, the Modbus TCP loop and the Home Assistant export of `pl-hems-mpc`."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hems_mpc.tariffs import (
    K_BANDS,
    TariffSpec,
    build_price_frame,
    compute_ci_k_factor,
    k_band,
    settle_prosumer_net_billing,
)
from hems_mpc.physics import ProsumerPlantSpec, simulate_rooftop_pv_kw, heat_pump_cop
from hems_mpc.controllers import solve_ci_day_milp, solve_day_milp
from hems_mpc.modbus_hil import VirtualHybridInverter, ModbusTCPServer, ModbusTCPClient, to_int16
from hems_mpc.ha_bridge import (
    build_mqtt_discovery_payloads,
    ev_current_limit_a,
    format_state_message,
    render_ha_modbus_yaml,
    sg_ready_from_dispatch,
)

REPO = Path(__file__).resolve().parents[1]


def _working_day(day: str) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    ts = pd.date_range(f"{day} 00:00", periods=96, freq="15min", tz="Europe/Warsaw").tz_convert("UTC")
    df_p = build_price_frame(ts, np.full(96, 400.0), np.full(96, 350.0), TariffSpec("C23", "CI_C23_DYNAMIC"))
    return ts, df_p


def test_net_billing_deposit_rules() -> None:
    """Negative RCE credits nothing, a credit is usable from the next month, refunds respect the cap."""
    ts = pd.DatetimeIndex(
        ["2026-05-10 10:00", "2026-05-10 11:00", "2026-05-20 20:00", "2026-06-10 20:00"], tz="UTC"
    )
    df_p = build_price_frame(ts, np.full(4, 300.0), np.array([400.0, -150.0, 300.0, 300.0]), TariffSpec("G11", "G11"))
    df_p["e_export_mwh"] = [1.0, 1.0, 0.0, 0.0]
    df_p["e_import_mwh"] = [0.0, 0.0, 0.1, 0.1]

    res = settle_prosumer_net_billing(df_p, refund_cap_share=0.30)
    spec = TariffSpec("G11", "G11")
    act = 0.1 * spec.energy_flat_pln_mwh
    assert res["export_credit_gross_pln"] == 400.0  # the -150 PLN/MWh quarter adds nothing
    assert res["distribution_cash_pln"] == pytest.approx(2 * 0.1 * spec.dist_flat_pln_mwh)
    assert res["active_energy_cash_pln"] == pytest.approx(act)  # May import precedes the credit's assignment
    assert res["deposit_used_pln"] == pytest.approx(act)  # June import is covered by the May credit
    assert res["deposit_refund_cash_pln"] <= 0.30 * 400.0 + 1e-9
    total = res["deposit_used_pln"] + res["deposit_refund_cash_pln"] + res["deposit_forfeited_pln"]
    assert total == pytest.approx(res["export_credit_gross_pln"])


def test_capacity_fee_rate_and_window() -> None:
    """The fee applies 07:00-21:59 on working days at the 2025 or 2026 rate, never on holidays."""
    ts = pd.date_range("2025-12-31 00:00", "2026-01-02 23:45", freq="15min", tz="Europe/Warsaw").tz_convert("UTC")
    df_p = build_price_frame(ts, np.full(len(ts), 400.0), np.full(len(ts), 350.0), TariffSpec("C23", "CI_C23_DYNAMIC"))
    loc = df_p["ts_local"]
    hour = loc.dt.hour
    rate = df_p["cap_fee_base_pln_mwh"]
    in_win = (hour >= 7) & (hour < 22)
    spec = TariffSpec("C23", "CI_C23_DYNAMIC")
    assert set(rate[(loc.dt.day == 31) & in_win]) == {spec.capacity_fee_2025_pln_mwh}
    assert set(rate[(loc.dt.day == 2) & in_win]) == {spec.capacity_fee_2026_pln_mwh}
    assert (rate[loc.dt.day == 1] == 0.0).all()  # 1 January is a public holiday
    assert (rate[~in_win] == 0.0).all()


@pytest.mark.parametrize("delta_pct", [0.0, 4.9, 5.1, 9.9, 10.1, 14.9, 15.1, 80.0])
def test_k_group_boundaries_and_fee(delta_pct: float) -> None:
    """A day's group follows the peak/off-peak difference and the fee is A x peak energy x rate."""
    ts, df_p = _working_day("2026-05-12")
    win = df_p["cap_window"].to_numpy()
    p_kw = np.where(win, 20.0 * (1.0 + delta_pct / 100.0), 20.0)
    e = p_kw * 0.25 / 1000.0
    k = compute_ci_k_factor(e, win, ts, df_p["cap_fee_base_pln_mwh"].to_numpy())
    a_coef, label = k_band(delta_pct)
    assert k["k_days"][label] == 1 and sum(k["k_days"].values()) == 1
    assert k["effective_k_multiplier"] == pytest.approx(a_coef, abs=1e-3)
    assert k["capacity_fee_actual_pln"] == pytest.approx(a_coef * e[win].sum() * 219.40, abs=0.01)


def test_k_group_zero_offpeak_is_k4() -> None:
    ts, df_p = _working_day("2026-05-12")
    win = df_p["cap_window"].to_numpy()
    e = np.where(win, 0.005, 0.0)
    k = compute_ci_k_factor(e, win, ts, df_p["cap_fee_base_pln_mwh"].to_numpy())
    assert k["k_days"]["K4"] == 1


def _prosumer_day(rce_noon: float) -> tuple[dict, dict]:
    T = 96
    spec = ProsumerPlantSpec()
    pv = np.zeros(T)
    pv[40:60] = 9.0
    rce = np.full(T, 300.0)
    rce[44:56] = rce_noon
    inputs = dict(
        dt_h=0.25,
        pv_avail_kw=pv,
        base_load_kw=np.full(T, 0.5),
        q_heat_req_kwth=np.full(T, 0.3),
        cop=heat_pump_cop(np.full(T, 12.0)),
        ev_plugged=np.arange(T) < 24,
        ev_target_kwh=8.0,
        p_buy_energy=np.full(T, 600.0),
        p_buy_dist=np.full(T, 250.0),
        p_sell_rce=np.maximum(0.0, rce),
        rce_raw=rce,
        soc_init_kwh=5.0,
        spec=spec,
    )
    return solve_day_milp(**inputs), inputs


@pytest.mark.parametrize("rce_noon", [-120.0, 3000.0])
def test_prosumer_milp_invariants(rce_noon: float) -> None:
    """Power balance, SoC bounds, EV energy, no export at RCE <= 0 and no simultaneous import and export."""
    sol, inp = _prosumer_day(rce_noon)
    spec = inp["spec"]
    net = (
        sol["p_imp_kw"] - sol["p_exp_kw"] - sol["p_ch_kw"] + sol["p_dis_kw"]
        - sol["p_curt_kw"] - sol["p_hp_kw"] - sol["p_ev_kw"]
    )
    np.testing.assert_allclose(net, inp["base_load_kw"] + spec.bess_standby_kw - inp["pv_avail_kw"], atol=1e-5)
    assert np.all(sol["soc_kwh"] >= spec.soc_min * spec.bess_energy_kwh - 1e-5)
    assert np.all(sol["soc_kwh"] <= spec.soc_max * spec.bess_energy_kwh + 1e-5)
    assert np.minimum(sol["p_imp_kw"], sol["p_exp_kw"]).max() < 1e-6
    assert np.all(sol["p_exp_kw"] + sol["p_curt_kw"] <= inp["pv_avail_kw"] + 1e-6)
    assert (sol["p_ev_kw"] * 0.25 * spec.ev_eta).sum() == pytest.approx(inp["ev_target_kwh"], abs=1e-4)
    assert sol["e_th_kwh"][-1] >= -1e-6
    if rce_noon <= 0.0:
        assert sol["p_exp_kw"][44:56].max() < 1e-4


def test_ci_milp_band_matches_settlement() -> None:
    """The MILP's chosen K-group equals the group the settlement computes from its own import profile."""
    ts, df_p = _working_day("2026-05-12")
    win = df_p["cap_window"].to_numpy()
    hour = df_p["ts_local"].dt.hour.to_numpy()
    load = np.where((hour >= 7) & (hour < 18), 38.0, 17.5)
    pv = np.zeros(96)
    sol = solve_ci_day_milp(
        dt_h=0.25,
        load_kw=load,
        pv_kw=pv,
        buy_pln_mwh=df_p["total_buy_no_cap_pln_mwh"].to_numpy(),
        rce_pln_mwh=df_p["rce_raw_pln_mwh"].to_numpy(),
        cap_window=win,
        cap_rate_pln_mwh=219.40,
        soc_init_kwh=50.0,
    )
    k = compute_ci_k_factor(sol["p_imp_kw"] * 0.25 / 1000.0, win, ts, df_p["cap_fee_base_pln_mwh"].to_numpy())
    assert k["k_days"][sol["band"]] == 1
    assert sol["band"] in {label for _, _, label in K_BANDS}
    assert np.minimum(sol["p_ch_kw"], sol["p_dis_kw"]).max() < 1e-6
    assert np.all(sol["p_imp_kw"] <= np.maximum(60.0, load) + 1e-6)
    assert sol["soc_kwh"][-1] >= 50.0 - 1e-6
    net = sol["p_imp_kw"] - sol["p_exp_kw"] - sol["p_ch_kw"] + sol["p_dis_kw"] - sol["p_curt_kw"]
    np.testing.assert_allclose(net, load - pv, atol=1e-5)


def test_ci_milp_flattens_when_battery_suffices() -> None:
    """With a small daytime excess the optimum is K1: shifting it costs less than the avoided fee."""
    ts, df_p = _working_day("2026-05-12")
    win = df_p["cap_window"].to_numpy()
    load = np.where(win, 22.0, 20.0)
    sol = solve_ci_day_milp(
        dt_h=0.25,
        load_kw=load,
        pv_kw=np.zeros(96),
        buy_pln_mwh=df_p["total_buy_no_cap_pln_mwh"].to_numpy(),
        rce_pln_mwh=df_p["rce_raw_pln_mwh"].to_numpy(),
        cap_window=win,
        cap_rate_pln_mwh=219.40,
        soc_init_kwh=50.0,
    )
    assert sol["band"] == "K1"


def test_modbus_tcp_roundtrip_and_watchdog_fallback() -> None:
    """FC 0x03/0x10 over a localhost socket; the emulated watchdog reverts to self-consumption."""
    inv = VirtualHybridInverter(bess_capacity_wh=10_000.0, max_charge_w=5_000, max_discharge_w=5_000)
    srv = ModbusTCPServer(inv, host="127.0.0.1", port=0)
    srv.start()
    try:
        cli = ModbusTCPClient(srv.host, srv.port)
        cli.connect()
        cli.write_multiple_registers(40001, [2, -3200, 0, 10])
        inv.step_physics(pv_avail_w=0, load_w=3500, dt_s=900)
        regs = cli.read_holding_registers(40010, 7)
        assert to_int16(regs[3]) == -3200
        assert regs[6] == 0

        inv.step_physics(pv_avail_w=6000, load_w=1000, dt_s=900)
        assert cli.read_holding_registers(40016, 1)[0] == 0  # 900 s is not more than the 900 s timeout
        inv.step_physics(pv_avail_w=6000, load_w=1000, dt_s=900)
        assert cli.read_holding_registers(40016, 1)[0] == 1
        assert cli.read_holding_registers(40001, 1)[0] == 0
        cli.close()
    finally:
        srv.stop()


def test_ha_export_consistency() -> None:
    """Every discovery template reads a key that the state message provides; mappings stay in range."""
    disc = build_mqtt_discovery_payloads("test_node")
    msg = json.loads(format_state_message(-2500, 0, -25.0, 10.0, 55.2, sg_ready="0:1", ev_current_a=16))
    assert msg["zero_export"] is True
    ids = [d["unique_id"] for d in disc.values()]
    assert len(ids) == len(set(ids))
    for d in disc.values():
        key = d["value_template"].split("value_json.")[1].split()[0]
        assert key in msg

    yaml_txt = render_ha_modbus_yaml("10.0.0.15", 502)
    assert "address: 40001" in yaml_txt and "10.0.0.15" in yaml_txt

    assert sg_ready_from_dispatch(2.0, 1.0, 3.5) == 3
    assert sg_ready_from_dispatch(3.5, 1.0, 3.5) == 4
    assert sg_ready_from_dispatch(0.0, 1.0, 3.5) == 1
    assert sg_ready_from_dispatch(1.05, 1.0, 3.5) == 2
    amps = [ev_current_limit_a(kw) for kw in np.linspace(0.0, 15.0, 61)]
    assert all(a == 0 or 6 <= a <= 16 for a in amps)
    assert amps == sorted(amps)


def test_pv_model_limits() -> None:
    ts = pd.date_range("2026-06-21 00:00", periods=96, freq="15min", tz="UTC")
    hour = np.asarray(ts.hour + ts.minute / 60.0)
    ghi = np.where((hour >= 3.0) & (hour < 19.0), 850.0, 0.0)
    pv_kw = simulate_rooftop_pv_kw(ts, ghi, ghi * 0.18, np.full(96, 24.0), np.full(96, 3.0), pv_kwp=10.0)
    assert np.all((pv_kw >= 0.0) & (pv_kw <= 10.0))
    assert pv_kw[ghi == 0.0].max() == 0.0
    assert pv_kw.max() > 5.0


def test_package_runs_as_module() -> None:
    assert importlib.util.find_spec("hems_mpc.__main__") is not None


def test_published_outputs_agree() -> None:
    """results/, docs/data.js and the README quote the same numbers."""
    summary = json.loads((REPO / "results" / "summary.json").read_text(encoding="utf-8"))
    data_js = (REPO / "docs" / "data.js").read_text(encoding="utf-8")
    assert json.loads(data_js.removeprefix("window.DATA = ").rstrip().rstrip(";")) == summary

    readme = (REPO / "README.md").read_text(encoding="utf-8")
    pros = pd.read_csv(REPO / "results" / "prosumer_summary.csv")
    for _, row in pros.iterrows():
        assert f"{row['total_economic_cost_pln']:,.0f}" in readme
    ci = pd.read_csv(REPO / "results" / "ci_c23_summary.csv", index_col="mode")
    for mode, row in ci.iterrows():
        assert f"{row['annual_total_net_cost_pln']:,.0f}" in readme
        assert summary["ci_summary"][mode]["annual_total_net_cost_pln"] == row["annual_total_net_cost_pln"]
