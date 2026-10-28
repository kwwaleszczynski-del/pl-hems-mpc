"""Unit and Hardware-in-the-Loop integration tests for `pl-hems-mpc`."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from hems_mpc.tariffs import (
    TariffSpec,
    build_price_frame,
    settle_prosumer_net_billing,
    compute_ci_k_factor,
)
from hems_mpc.physics import ProsumerPlantSpec, simulate_rooftop_pv_kw, heat_pump_cop
from hems_mpc.controllers import solve_day_milp
from hems_mpc.modbus_hil import VirtualHybridInverter, ModbusTCPServer, ModbusTCPClient, to_int16
from hems_mpc.ha_bridge import build_mqtt_discovery_payloads, render_ha_modbus_yaml, format_state_message
from hems_mpc.hardware_profiles import (
    INVERTER_PROFILES,
    compute_sg_ready_state,
    compute_ocpp_ev_current_a,
    render_vendor_ha_yaml,
)
from hems_mpc.vpp_fleet import simulate_vpp_fleet_benchmark
from custom_components.pl_hems_mpc.sensor import build_live_ha_entity_states


def test_net_billing_negative_rce_and_20pct_refund_cap() -> None:
    """Verify 0 PLN deposit credit at negative RCE, cash-only distribution, and 20% refund cap."""
    ts = pd.date_range("2026-05-01 00:00:00", periods=4, freq="15min", tz="UTC")
    sdac = np.array([200.0, 200.0, -80.0, 500.0])
    rce = np.array([200.0, -150.0, -50.0, 400.0])
    df_p = build_price_frame(ts, sdac, rce, TariffSpec(name="G11", kind="G11"))

    # Step 1: export 1 MWh at RCE=-150 -> should give 0 PLN credit
    # Step 3: export 1 MWh at RCE=+400 -> should give 400 PLN credit
    # Step 0: import 0.1 MWh -> active energy = 62 PLN (covered by deposit), dist = 29.5 PLN (paid in cash!)
    df_p["e_export_mwh"] = np.array([0.0, 1.0, 0.0, 1.0])
    df_p["e_import_mwh"] = np.array([0.1, 0.0, 0.0, 0.0])

    res = settle_prosumer_net_billing(df_p, refund_cap_share=0.20)
    assert res["export_credit_gross_pln"] == 400.0
    assert res["deposit_used_pln"] == 62.0
    assert res["distribution_cash_pln"] == 29.5
    # Remaining deposit is 400 - 62 = 338 PLN, but refund is capped at 20% of 400 = 80 PLN!
    assert res["deposit_refund_cash_pln"] == 80.0
    assert res["deposit_forfeited_80pct_pln"] == 258.0


def test_ci_capacity_fee_k1_vs_k4() -> None:
    """Verify K1 (83% discount, A=0.17) for flat load vs K4 (A=1.00) for daytime-peaking load."""
    ts = pd.date_range("2026-05-11 00:00:00", periods=96 * 5, freq="15min", tz="Europe/Warsaw").tz_convert("UTC")
    df_p = build_price_frame(
        ts,
        np.full(len(ts), 400.0),
        np.full(len(ts), 350.0),
        TariffSpec(name="B23", kind="CI_B23_DYNAMIC"),
    )
    cap_win = df_p["cap_window"].to_numpy()

    # Case 1: flat 20 kW import all day & night -> delta = 0% < 5% -> K1 (A = 0.17)
    e_flat = np.full(len(ts), 20.0 * 0.25 / 1000.0)
    k1 = compute_ci_k_factor(e_flat, cap_win, ts)
    assert k1["effective_k_multiplier"] == 0.17
    assert k1["k1_days_share_pct"] == 100.0

    # Case 2: 50 kW in peak hours, 10 kW in off-peak -> delta = +400% >= 15% -> K4 (A = 1.00)
    e_peaky = np.where(cap_win, 50.0, 10.0) * 0.25 / 1000.0
    k4 = compute_ci_k_factor(e_peaky, cap_win, ts)
    assert k4["effective_k_multiplier"] == 1.00
    assert k4["k1_days_share_pct"] == 0.0


def test_milp_day_power_balance_and_negative_rce_curtailment() -> None:
    """Ensure MILP maintains exact AC bus balance, SoC limits, and curtails PV when RCE <= 0."""
    T = 96
    dt_h = 0.25
    spec = ProsumerPlantSpec()
    pv = np.zeros(T)
    pv[40:60] = 9.0  # Strong midday solar
    base_load = np.full(T, 0.5)
    q_heat = np.full(T, 0.3)
    cop = heat_pump_cop(np.full(T, 12.0))
    ev_plug = np.zeros(T, dtype=bool)
    ev_plug[0:24] = True

    p_buy_e = np.full(T, 600.0)
    p_buy_d = np.full(T, 250.0)
    rce_raw = np.full(T, 300.0)
    rce_raw[44:56] = -120.0  # 3 hours of negative RCE at solar noon
    p_sell = np.maximum(0.0, rce_raw)

    sol = solve_day_milp(
        dt_h=dt_h,
        pv_avail_kw=pv,
        base_load_kw=base_load,
        q_heat_req_kwth=q_heat,
        cop=cop,
        ev_plugged=ev_plug,
        ev_target_kwh=8.0,
        p_buy_energy=p_buy_e,
        p_buy_dist=p_buy_d,
        p_sell_rce=p_sell,
        rce_raw=rce_raw,
        soc_init_kwh=5.0,
        spec=spec,
    )

    # Check power balance at every 15-min step
    net_lhs = (
        sol["p_imp_kw"]
        - sol["p_exp_kw"]
        - sol["p_ch_kw"]
        + sol["p_dis_kw"]
        - sol["p_curt_kw"]
        - sol["p_hp_kw"]
        - sol["p_ev_kw"]
    )
    net_rhs = base_load + spec.bess_standby_kw - pv
    np.testing.assert_allclose(net_lhs, net_rhs, atol=1e-5)

    # Check SoC bounds
    assert np.all(sol["soc_kwh"] >= spec.soc_min * spec.bess_energy_kwh - 1e-5)
    assert np.all(sol["soc_kwh"] <= spec.soc_max * spec.bess_energy_kwh + 1e-5)

    # During negative RCE window (44:56), export must be zero!
    assert np.max(sol["p_exp_kw"][44:56]) < 1e-4


def test_modbus_tcp_hil_roundtrip_and_watchdog_fallback() -> None:
    """Verify real TCP socket FC 0x03/0x10 and hardware watchdog fallback on lost EMS heartbeat."""
    inv = VirtualHybridInverter(bess_capacity_wh=10_000.0, max_charge_w=5_000, max_discharge_w=5_000)
    srv = ModbusTCPServer(inv, host="127.0.0.1", port=0)
    srv.start()
    try:
        cli = ModbusTCPClient(srv.host, srv.port)
        cli.connect()

        # 1. Command Forced Discharge (-3,200 W), zero-export limit = 0 W, heartbeat = 10
        cli.write_multiple_registers(40001, [2, -3200, 0, 10])
        inv.step_physics(pv_avail_w=0, load_w=3500, dt_s=900)

        regs = cli.read_holding_registers(40010, 7)
        assert to_int16(regs[3]) == -3200  # actual battery power = -3200 W
        assert regs[6] == 0                # watchdog OK

        # 2. Now stop updating heartbeat for 2 x 900s = 1800s (> 900s timeout in reg 40005)
        inv.step_physics(pv_avail_w=6000, load_w=1000, dt_s=900)
        inv.step_physics(pv_avail_w=6000, load_w=1000, dt_s=900)

        regs_after = cli.read_holding_registers(40010, 7)
        mode_reg = cli.read_holding_registers(40001, 1)[0]
        assert regs_after[6] == 1  # watchdog tripped flag raised!
        assert mode_reg == 0       # reverted to safe local self-consumption!
        cli.close()
    finally:
        srv.stop()


def test_ha_bridge_and_pv_physics() -> None:
    """Verify Home Assistant MQTT discovery payloads, YAML snippet, and rooftop PV generator."""
    disc = build_mqtt_discovery_payloads("test_node")
    assert "homeassistant/sensor/test_node/bess_target_power/config" in disc
    yaml_txt = render_ha_modbus_yaml("10.0.0.15", 502)
    assert "address: 40001" in yaml_txt and "10.0.0.15" in yaml_txt
    msg = json.loads(format_state_message(-2500, 0, -25.0, 10.0, 55.2))
    assert msg["zero_export"] is True

    ts = pd.date_range("2026-06-21 10:00:00", periods=4, freq="15min", tz="UTC")
    pv_kw = simulate_rooftop_pv_kw(
        ts,
        ghi_wm2=np.full(4, 850.0),
        dhi_wm2=np.full(4, 150.0),
        temp_air_c=np.full(4, 24.0),
        wind_ms=np.full(4, 3.0),
        pv_kwp=10.0,
    )
    assert np.all((pv_kw > 5.0) & (pv_kw <= 10.0))


def test_multibrand_profiles_hacs_entities_and_vpp_fleet() -> None:
    """Verify Deye/Huawei/Sungrow/Victron profiles, HACS entity coordinator, and VPP P90 aggregation."""
    assert set(INVERTER_PROFILES.keys()) == {
        "deye_sg04lp3",
        "huawei_sun2000",
        "sungrow_sh10rt",
        "victron_cerbo_ess",
    }
    deye_yaml = render_vendor_ha_yaml("deye_sg04lp3", "192.168.1.88", 502)
    assert "deye_sg04lp3_hub" in deye_yaml and "192.168.1.88" in deye_yaml

    sg4 = compute_sg_ready_state(rce_pln_mwh=-120.0, total_buy_pln_mwh=150.0, pv_surplus_kw=4.5, th_buffer_kwh=1.0)
    assert sg4["state_code"] == "1:1" and sg4["mode_num"] == 4
    assert compute_ocpp_ev_current_a(11.0) == 16
    assert compute_ocpp_ev_current_a(0.0) == 0

    entities = build_live_ha_entity_states(
        ts_local_iso="2026-05-01T12:00:00+02:00",
        rce_pln_mwh=-450.0,
        sdac_pln_mwh=-80.0,
        total_buy_pln_mwh=95.0,
        pv_avail_kw=8.4,
        base_load_kw=0.6,
        bess_cmd_kw=5.0,
        soc_pct=78.0,
        ev_cmd_kw=7.6,
        th_buf_kwh=2.1,
        fifo_deposit_balance_pln=1120.0,
        fifo_at_risk_80pct_pln=340.0,
        ci_delta_pct=3.4,
        inverter_brand="deye_sg04lp3",
    )
    assert entities["binary_sensor.pl_hems_zero_export_guard"]["state"] == "ON"
    assert entities["sensor.pl_hems_ci_k_factor_live"]["state"] == "K1 (83% Discount)"
    assert entities["sensor.pl_hems_ocpp_ev_current_a"]["state"] == 11

    # VPP benchmark check on synthetic 96-step frame
    df_dummy = pd.DataFrame(
        {
            "soc_kwh": np.full(96, 5.5),
            "p_ch_kw": np.zeros(96),
            "p_dis_kw": np.zeros(96),
            "rce_raw_pln_mwh": np.full(96, 320.0),
            "sdac_pln_mwh": np.full(96, 410.0),
        }
    )
    vpp = simulate_vpp_fleet_benchmark(df_dummy, n_homes=250, n_ci_sites=15)
    assert vpp["p90_firm_upward_mw"] >= 1.0
    assert vpp["p90_firm_downward_mw"] >= 1.0
    assert vpp["vpp_bonus_per_prosumer_home_pln_yr"] > 200

