"""End-to-end simulation pipeline, Modbus TCP HIL verification, plotting, and dashboard generation."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .tariffs import TariffSpec, build_price_frame, settle_prosumer_net_billing
from .physics import ProsumerPlantSpec, simulate_rooftop_pv_kw, generate_household_profiles
from .controllers import run_prosumer_strategy, simulate_ci_site
from .modbus_hil import VirtualHybridInverter, ModbusTCPServer, ModbusTCPClient, to_int16
from .ha_bridge import build_mqtt_discovery_payloads, render_ha_modbus_yaml, format_state_message
from .hardware_profiles import (
    INVERTER_PROFILES,
    compute_sg_ready_state,
    compute_ocpp_ev_current_a,
    render_vendor_ha_yaml,
)
from .vpp_fleet import simulate_vpp_fleet_benchmark


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_PRICE_DIR = REPO_ROOT.parent / "pl-capture-price" / "data"
EXTERNAL_WEATHER_PARQUET = (
    REPO_ROOT.parent
    / "cable-pooling-sizer"
    / "data"
    / "weather"
    / "era5_central_52.75_18.25_20200101_20260930.parquet"
)


def load_market_and_weather(
    start_utc: str = "2025-10-01 00:00:00+00:00",
    end_utc: str = "2026-09-30 23:45:00+00:00",
) -> tuple[pd.DataFrame, str]:
    """Load 15-minute Polish SDAC + RCE prices and ERA5 weather for 2025-10..2026-09.

    Uses cached real PSE/TGE & ERA5 parquet files if available on disk; otherwise falls back
    to a realistic calibrated synthetic series so CI tests run anywhere without external data.
    """
    ts_utc = pd.date_range(start_utc, end_utc, freq="15min", tz="UTC")
    local_cache = REPO_ROOT / "data" / "pl_hems_inputs_2025_2026.parquet"

    if local_cache.exists():
        df = pd.read_parquet(local_cache)
        return df.reindex(ts_utc).interpolate().bfill().ffill(), "Real PSE RCE + SDAC + ERA5 (cached)"

    if EXTERNAL_PRICE_DIR.exists() and EXTERNAL_WEATHER_PARQUET.exists():
        sdac_parts = sorted(EXTERNAL_PRICE_DIR.glob("sdac_*.parquet"))
        rce_parts = sorted(EXTERNAL_PRICE_DIR.glob("rce_*.parquet"))
        df_sdac = pd.concat([pd.read_parquet(p) for p in sdac_parts]).sort_index()
        df_sdac = df_sdac[~df_sdac.index.duplicated(keep="last")]
        df_rce = pd.concat([pd.read_parquet(p) for p in rce_parts]).sort_index()
        df_rce = df_rce[~df_rce.index.duplicated(keep="last")]

        df_wx = pd.read_parquet(EXTERNAL_WEATHER_PARQUET).sort_index()
        df_wx = df_wx[~df_wx.index.duplicated(keep="last")]

        out = pd.DataFrame(index=ts_utc)
        out["sdac_pln_mwh"] = df_sdac["price"].reindex(ts_utc).interpolate().bfill().ffill()
        out["rce_pln_mwh"] = df_rce["price"].reindex(ts_utc).interpolate().bfill().ffill()

        wx_15 = df_wx[["ghi", "dhi", "temp_air", "wind_10m"]].reindex(ts_utc).interpolate(method="time").bfill().ffill()
        out["ghi_wm2"] = np.maximum(0.0, wx_15["ghi"].to_numpy())
        out["dhi_wm2"] = np.maximum(0.0, wx_15["dhi"].to_numpy())
        out["temp_air_c"] = wx_15["temp_air"].to_numpy()
        out["wind_ms"] = np.maximum(0.2, wx_15["wind_10m"].to_numpy())

        local_cache.parent.mkdir(parents=True, exist_ok=True)
        out.astype("float32").to_parquet(local_cache)
        return out, "Real PSE RCE + TGE SDAC + ERA5 Central Poland (2025-10 to 2026-09)"

    # Synthetic fallback for offline CI
    rng = np.random.default_rng(2026)
    n = len(ts_utc)
    ts_local = ts_utc.tz_convert("Europe/Warsaw")
    doy = np.asarray(ts_local.day_of_year, dtype=float)
    hour = np.asarray(ts_local.hour + ts_local.minute / 60.0, dtype=float)

    temp_air_c = 9.0 - 10.5 * np.cos(2 * np.pi * (doy - 20.0) / 365.0) + 3.8 * np.sin(2 * np.pi * (hour - 9.0) / 24.0)
    solar_env = np.maximum(0.0, np.sin(np.pi * (hour - 5.5) / 13.5))
    season_irr = 0.25 + 0.75 * np.sin(np.pi * np.clip((doy - 10.0) / 345.0, 0.0, 1.0)) ** 1.3
    cloud = np.clip(rng.beta(2.2, 1.5, size=n), 0.1, 1.0)
    ghi_wm2 = 880.0 * solar_env * season_irr * cloud
    dhi_wm2 = np.minimum(ghi_wm2, ghi_wm2 * (0.32 + 0.45 * (1.0 - cloud)))
    wind_ms = np.clip(rng.weibull(2.0, size=n) * 3.5, 0.5, 14.0)

    duck_dip = -260.0 * (ghi_wm2 / 800.0)
    eve_spike = 310.0 * np.exp(-0.5 * ((hour - 19.5) / 1.8) ** 2)
    sdac = 430.0 + duck_dip + eve_spike + rng.normal(0.0, 45.0, size=n)
    rce = sdac - 35.0 * (ghi_wm2 / 700.0) + rng.normal(0.0, 35.0, size=n)

    out = pd.DataFrame(
        {
            "sdac_pln_mwh": sdac,
            "rce_pln_mwh": rce,
            "ghi_wm2": ghi_wm2,
            "dhi_wm2": dhi_wm2,
            "temp_air_c": temp_air_c,
            "wind_ms": wind_ms,
        },
        index=ts_utc,
    )
    return out, "Synthetic Calibrated Polish 15-min Dataset"


def run_modbus_hil_verification(
    df_dispatch: pd.DataFrame,
    start_date: str = "2026-04-29",
    days: int = 7,
) -> pd.DataFrame:
    """Execute a closed-loop 7-day Hardware-in-the-Loop test over a real localhost Modbus TCP socket.

    Includes a simulated 45-minute Home Assistant / Wi-Fi communication drop on Day 3 (steps 240..242)
    to verify that the hybrid inverter's hardware watchdog trips after 15 minutes (`register 40005 = 900s`)
    and reverts `register 40001` from Forced EMS (`2`) to Local Self-Consumption (`0`).
    """
    ts_local = df_dispatch["ts_local"]
    mask = (ts_local >= pd.Timestamp(start_date, tz="Europe/Warsaw")) & (
        ts_local < pd.Timestamp(start_date, tz="Europe/Warsaw") + pd.Timedelta(days=days)
    )
    sub = df_dispatch.loc[mask].copy()
    if len(sub) < 96:
        sub = df_dispatch.iloc[: 96 * days].copy()

    inv = VirtualHybridInverter(bess_capacity_wh=10_000.0, max_charge_w=5_000, max_discharge_w=5_000)
    server = ModbusTCPServer(inv, host="127.0.0.1", port=0)
    server.start()

    records = []
    try:
        client = ModbusTCPClient(host=server.host, port=server.port, unit_id=1)
        client.connect()

        for step_i, (ts_idx, row) in enumerate(sub.iterrows()):
            pv_w = int(round(float(row["pv_avail_kw"]) * 1000.0))
            load_w = int(round(float(row["base_load_kw"] + row["p_hp_kw"] + row["p_ev_kw"] + 0.035) * 1000.0))
            planned_batt_w = int(round(float(row["p_ch_kw"] - row["p_dis_kw"]) * 1000.0))
            rce_now = float(row["rce_raw_pln_mwh"])
            exp_limit_w = 0 if rce_now <= 0.0 else 10_000

            # Simulate a 45-minute EMS / Wi-Fi drop on Day 3 around solar noon (steps 240, 241, 242)
            comms_lost = 240 <= step_i <= 242
            if not comms_lost:
                hb = (step_i + 1) & 0xFFFF
                # Write registers 40001..40004 in a single FC 0x10 transaction
                client.write_multiple_registers(40001, [2, planned_batt_w, exp_limit_w, hb])

            # Step physical inverter by 15 minutes (900 seconds)
            inv.step_physics(pv_avail_w=pv_w, load_w=load_w, dt_s=900)

            # Read back telemetry registers 40010..40016 via FC 0x03
            regs = client.read_holding_registers(40010, 7)
            soc_pct = regs[0] / 10.0
            pv_meas_w = regs[1]
            load_meas_w = regs[2]
            batt_meas_w = to_int16(regs[3])
            grid_meas_w = to_int16(regs[4])
            curt_meas_w = regs[5]
            wd_tripped = regs[6]

            records.append(
                {
                    "timestamp_utc": ts_idx,
                    "timestamp_local": row["ts_local"],
                    "rce_pln_mwh": round(rce_now, 2),
                    "sdac_pln_mwh": round(float(row["sdac_pln_mwh"]), 2),
                    "pv_avail_w": pv_meas_w,
                    "load_total_w": load_meas_w,
                    "cmd_batt_w": planned_batt_w,
                    "actual_batt_w": batt_meas_w,
                    "grid_pcc_w": grid_meas_w,
                    "curtailed_pv_w": curt_meas_w,
                    "soc_pct": round(soc_pct, 1),
                    "comms_lost": int(comms_lost),
                    "watchdog_tripped": int(wd_tripped),
                }
            )

        client.close()
    finally:
        server.stop()

    return pd.DataFrame(records)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=135, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def generate_plots(
    df_prosumer_summary: pd.DataFrame,
    df_dyn_mpc: pd.DataFrame,
    df_dyn_unc: pd.DataFrame,
    ci_results: dict[str, dict[str, float]],
    df_hil: pd.DataFrame,
    vpp_summary: dict,
    ha_24h_frames: list[dict],
    img_dir: Path,
) -> None:
    """Write README charts in the portfolio's light paper style."""
    img_dir.mkdir(parents=True, exist_ok=True)
    INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
    BLUE, NAVY, ORANGE, WARM, GREEN, RED = "#2a78d6", "#1b3f8f", "#eb6834", "#e8a33d", "#1baf7a", "#c8423b"

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "normal",
            "axes.grid": True,
            "grid.alpha": 0.45,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
            "grid.color": GRID,
        }
    )

    def _style_ax(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)

    # 1. Prosumer annual cost by tariff and control strategy
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    tariffs = ["G11", "G12w", "DYNAMIC_G12w"]
    strategies = [
        ("uncontrolled", "Uncontrolled", MUTED),
        ("neg_price_guard", "Negative-price guard", WARM),
        ("ha_heuristic", "Time-window rules (YAML)", ORANGE),
        ("mpc_forecast", "MILP, forecast (realizable)", BLUE),
        ("milp_oracle", "MILP, perfect foresight", NAVY),
    ]
    x = np.arange(len(tariffs))
    w = 0.15
    for i, (st_code, st_label, color) in enumerate(strategies):
        vals = [
            float(
                df_prosumer_summary[
                    (df_prosumer_summary["tariff"] == tf) & (df_prosumer_summary["strategy"] == st_code)
                ]["total_economic_cost_pln"].iloc[0]
            )
            for tf in tariffs
        ]
        bars = ax.bar(x + (i - 2) * w, vals, width=w, label=st_label, color=color, edgecolor=SURFACE, linewidth=0.8)
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                h + 70,
                f"{int(round(h)):,}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                color=INK2,
            )
    _style_ax(ax)
    ax.set_xticks(x, ["G11 (flat)", "G12w (two-zone)", "Dynamic SDAC + G12w"])
    ax.set_ylabel("Annual net bill + battery wear [PLN / yr]")
    ax.set_title("Prosumer (10 kWp PV, 10 kWh battery, heat pump, EV): annual cost, Oct 2025 - Sep 2026")
    ax.set_ylim(0, max(df_prosumer_summary["total_economic_cost_pln"]) * 1.22)
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    _save(fig, img_dir / "bill_comparison.png")

    # 2. Example week dispatch (1-3 May 2026)
    fig, axes = plt.subplots(3, 1, figsize=(9.6, 6.2), sharex=True)
    ts_l = df_dyn_mpc["ts_local"]
    win = (ts_l >= pd.Timestamp("2026-05-01 00:00:00", tz="Europe/Warsaw")) & (
        ts_l < pd.Timestamp("2026-05-04 00:00:00", tz="Europe/Warsaw")
    )
    if win.sum() < 96:
        win = slice(0, 96 * 3)
    s_mpc = df_dyn_mpc.loc[win]
    s_unc = df_dyn_unc.loc[win]
    t_axis = s_mpc["ts_local"]

    axes[0].plot(
        t_axis,
        s_mpc["energy_buy_pln_mwh"] + s_mpc["dist_buy_pln_mwh"],
        label="Import price (SDAC + G12w)",
        color=INK2,
        lw=1.3,
    )
    axes[0].plot(t_axis, s_mpc["rce_raw_pln_mwh"], label="RCE export price", color=BLUE, lw=1.3)
    axes[0].axhline(0.0, color=MUTED, ls="--", lw=0.8)
    axes[0].fill_between(
        t_axis,
        s_mpc["rce_raw_pln_mwh"],
        0.0,
        where=(s_mpc["rce_raw_pln_mwh"] <= 0.0),
        color=ORANGE,
        alpha=0.22,
        label="RCE <= 0 (0 PLN credit)",
    )
    _style_ax(axes[0])
    axes[0].set_ylabel("PLN / MWh")
    axes[0].set_title("Dispatch on 1-3 May 2026 (Dynamic SDAC + G12w): negative midday RCE and evening peaks")
    axes[0].legend(loc="upper right", ncol=3, frameon=False, fontsize=7.8)

    axes[1].plot(t_axis, s_mpc["pv_avail_kw"], label="Available PV", color=WARM, lw=1.2)
    axes[1].plot(
        t_axis,
        s_mpc["p_ch_kw"] - s_mpc["p_dis_kw"],
        label="Battery (+charge / -discharge)",
        color=BLUE,
        lw=1.4,
    )
    axes[1].plot(t_axis, s_mpc["p_hp_kw"] + s_mpc["p_ev_kw"], label="Heat pump + EV", color=GREEN, lw=1.2)
    axes[1].fill_between(t_axis, 0, s_mpc["p_curt_kw"], color=RED, alpha=0.28, label="Curtailed PV")
    _style_ax(axes[1])
    axes[1].set_ylabel("Power [kW]")
    axes[1].legend(loc="upper right", ncol=4, frameon=False, fontsize=7.8)

    axes[2].plot(t_axis, s_unc["soc_kwh"] * 10.0, label="Uncontrolled", color=MUTED, ls="--", lw=1.2)
    axes[2].plot(t_axis, s_mpc["soc_kwh"] * 10.0, label="MILP, forecast (realizable)", color=BLUE, lw=1.6)
    _style_ax(axes[2])
    axes[2].set_ylabel("SoC [%]")
    axes[2].set_ylim(0, 105)
    axes[2].legend(loc="upper right", ncol=2, frameon=False, fontsize=8)
    _save(fig, img_dir / "dispatch_week.png")

    # 3. C&I B23 site cost breakdown & K-factor
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    modes = ["no_bess", "uncontrolled_bess", "mpc_ci_ems"]
    labels = ["PV only", "PV + battery\n(self-consumption)", "PV + battery\n(MILP, forecast)"]
    e_cost = [ci_results[m]["annual_energy_dist_pln"] / 1000.0 for m in modes]
    cap_cost = [ci_results[m]["annual_capacity_fee_pln"] / 1000.0 for m in modes]
    pen_cost = [ci_results[m]["annual_exceed_penalty_pln"] / 1000.0 for m in modes]
    net_tot = [ci_results[m]["annual_total_net_cost_pln"] / 1000.0 for m in modes]

    x_ci = np.arange(len(modes))
    ax1.bar(x_ci, e_cost, width=0.46, label="Energy + B23 distribution", color=BLUE)
    ax1.bar(x_ci, cap_cost, width=0.46, bottom=e_cost, label="PSE capacity fee", color=WARM)
    ax1.bar(
        x_ci,
        pen_cost,
        width=0.46,
        bottom=np.array(e_cost) + np.array(cap_cost),
        label="Contracted power penalty (>40 kW)",
        color=RED,
    )
    for i, tot in enumerate(net_tot):
        ax1.text(
            i,
            e_cost[i] + cap_cost[i] + pen_cost[i] + 2.5,
            f"net {tot:.1f} k",
            ha="center",
            fontsize=8,
            color=INK,
        )
    _style_ax(ax1)
    ax1.set_xticks(x_ci, labels)
    ax1.set_ylabel("Thousand PLN / yr")
    ax1.set_title("C&I site (50 kWp PV, 100 kWh battery, B23)")
    ax1.set_ylim(0, max(np.array(e_cost) + np.array(cap_cost) + np.array(pen_cost)) * 1.28)
    ax1.legend(loc="upper right", frameon=False, fontsize=7.5)

    k1_shares = [ci_results[m]["k1_days_share_pct"] for m in modes]
    k_mults = [ci_results[m]["effective_k_mult"] * 100.0 for m in modes]
    b1 = ax2.bar(x_ci - 0.16, k1_shares, width=0.30, label="Working days in K1 (<5 % peak/off-peak)", color=GREEN)
    b2 = ax2.bar(x_ci + 0.16, k_mults, width=0.30, label="Mean capacity-fee factor A [%]", color=ORANGE)
    for b in b1:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.0f}%", ha="center", fontsize=7.8)
    for b in b2:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.0f}%", ha="center", fontsize=7.8)
    _style_ax(ax2)
    ax2.set_xticks(x_ci, labels)
    ax2.set_ylabel("Share of days / factor A [%]")
    ax2.set_ylim(0, 118)
    ax2.set_title("PSE capacity fee K-factor (Art. 70a)")
    ax2.legend(loc="upper right", frameon=False, fontsize=7.5)
    _save(fig, img_dir / "ci_k_factor.png")

    # 4. Modbus TCP closed-loop trace
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(9.4, 4.6), sharex=True)
    sub_hil = df_hil.iloc[192:288]
    t_h = sub_hil["timestamp_local"]
    ax_a.plot(t_h, sub_hil["cmd_batt_w"] / 1000.0, label="Commanded battery (reg 40002)", color=INK2, ls="--", lw=1.2)
    ax_a.plot(t_h, sub_hil["actual_batt_w"] / 1000.0, label="Executed battery (reg 40013)", color=BLUE, lw=1.5)
    ax_a.plot(t_h, sub_hil["curtailed_pv_w"] / 1000.0, label="Curtailed PV (reg 40015)", color=ORANGE, lw=1.3)
    _style_ax(ax_a)
    ax_a.set_ylabel("Power [kW]")
    ax_a.set_title("Closed-loop Modbus TCP trace on 1 May 2026 with a 45-minute communication drop")
    ax_a.legend(loc="upper right", frameon=False, fontsize=7.8, ncol=3)

    ax_b.plot(t_h, sub_hil["soc_pct"], label="Inverter SoC (reg 40010)", color=BLUE, lw=1.5)
    ax_b.fill_between(
        t_h,
        0,
        100,
        where=(sub_hil["comms_lost"] > 0),
        color=RED,
        alpha=0.18,
        label="No heartbeat from EMS (45 min)",
    )
    ax_b.fill_between(
        t_h,
        0,
        100,
        where=(sub_hil["watchdog_tripped"] > 0),
        color=WARM,
        alpha=0.35,
        label="Watchdog tripped (reg 40016 = 1 -> local self-consumption)",
    )
    _style_ax(ax_b)
    ax_b.set_ylabel("SoC [%]")
    ax_b.set_ylim(0, 105)
    ax_b.legend(loc="upper right", frameon=False, fontsize=7.8, ncol=3)
    _save(fig, img_dir / "modbus_hil_loop.png")

    # 5. Home Assistant entities & aggregated fleet reserve
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    hours_24 = np.arange(len(ha_24h_frames)) * 0.25
    sg_modes = [f["sg_mode"] for f in ha_24h_frames]
    ev_amps = [f["ev_amps"] for f in ha_24h_frames]
    bess_kw_24 = [f["bess_w"] / 1000.0 for f in ha_24h_frames]

    ax_l.step(hours_24, ev_amps, where="post", label="Wallbox current [A] (OCPP)", color=BLUE, lw=1.4)
    ax_l.plot(hours_24, bess_kw_24, label="Battery setpoint [kW]", color=GREEN, lw=1.3)
    ax_l.step(hours_24, sg_modes, where="post", label="Heat pump SG Ready state (1-4)", color=ORANGE, lw=1.3, ls="--")
    _style_ax(ax_l)
    ax_l.set_xlabel("Hour on 1 May 2026")
    ax_l.set_ylabel("A / kW / state")
    ax_l.set_title("Home Assistant entities on 1 May 2026")
    ax_l.legend(loc="upper left", frameon=False, fontsize=7.5)

    cats = ["Installed\npower", "P50 upward", "P90 upward", "P90 downward"]
    mw_vals = [
        vpp_summary["installed_bess_power_mw"],
        vpp_summary["p50_upward_mw"],
        vpp_summary["p90_firm_upward_mw"],
        vpp_summary["p90_firm_downward_mw"],
    ]
    bars_v = ax_r.bar(np.arange(len(cats)), mw_vals, width=0.48, color=[MUTED, WARM, BLUE, NAVY])
    for b in bars_v:
        h = b.get_height()
        ax_r.text(b.get_x() + b.get_width() / 2, h + 0.04, f"{h:.2f}", ha="center", fontsize=8, color=INK2)
    ax_r.axhline(1.0, color=RED, ls="--", lw=1.0, label="1.0 MW bid threshold")
    _style_ax(ax_r)
    ax_r.set_xticks(np.arange(len(cats)), cats)
    ax_r.set_ylabel("MW")
    ax_r.set_ylim(0, max(mw_vals) * 1.24)
    ax_r.set_title("Aggregated reserve (250 homes + 15 C&I sites)")
    ax_r.legend(loc="upper right", frameon=False, fontsize=7.5)
    _save(fig, img_dir / "ha_vpp_overview.png")


def generate_dashboard_html(summary_payload: dict, docs_dir: Path) -> None:
    """Write docs/data.js for the static dashboard in docs/index.html."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / ".nojekyll").write_text("", encoding="utf-8")
    (docs_dir / "data.js").write_text(
        "window.DATA = " + json.dumps(summary_payload, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )


def run_full_pipeline(output_root: Path | None = None) -> dict:
    """Run all prosumer & C&I simulations, Modbus TCP HIL verification, VPP benchmark, plots, and HTML generation."""
    root = output_root or REPO_ROOT
    results_dir = root / "results"
    docs_dir = root / "docs"
    img_dir = docs_dir / "img"
    results_dir.mkdir(parents=True, exist_ok=True)

    df_in, source_label = load_market_and_weather()
    ts_utc = df_in.index
    spec = ProsumerPlantSpec()

    pv_avail_kw = simulate_rooftop_pv_kw(
        ts_utc=ts_utc,
        ghi_wm2=df_in["ghi_wm2"].to_numpy(),
        dhi_wm2=df_in["dhi_wm2"].to_numpy(),
        temp_air_c=df_in["temp_air_c"].to_numpy(),
        wind_ms=df_in["wind_ms"].to_numpy(),
        pv_kwp=spec.pv_kwp,
        inverter_kw=spec.pv_inverter_kw,
    )
    df_load = generate_household_profiles(
        ts_utc=ts_utc,
        temp_air_c=df_in["temp_air_c"].to_numpy(),
        ghi_wm2=df_in["ghi_wm2"].to_numpy(),
    )

    tariff_specs = [
        TariffSpec(name="G11 Flat", kind="G11"),
        TariffSpec(name="G12w Weekend", kind="G12w"),
        TariffSpec(name="Dynamic SDAC + G12w", kind="DYNAMIC_G12w"),
    ]
    strategies = ["uncontrolled", "neg_price_guard", "ha_heuristic", "mpc_forecast", "milp_oracle"]

    prosumer_rows = []
    saved_runs: dict[tuple[str, str], pd.DataFrame] = {}

    for tf_spec in tariff_specs:
        df_prices = build_price_frame(
            ts_utc=ts_utc,
            sdac_pln_mwh=df_in["sdac_pln_mwh"].to_numpy(),
            rce_pln_mwh=df_in["rce_pln_mwh"].to_numpy(),
            tariff=tf_spec,
        )
        unc_neg_export = 0.0
        for st in strategies:
            sim_df = run_prosumer_strategy(
                strategy=st,
                df_prices=df_prices,
                pv_avail_kw=pv_avail_kw,
                df_load=df_load,
                temp_out_c=df_in["temp_air_c"].to_numpy(),
                spec=spec,
            )
            if tf_spec.kind == "DYNAMIC_G12w" and st in ("uncontrolled", "mpc_forecast"):
                saved_runs[(tf_spec.kind, st)] = sim_df

            bill = settle_prosumer_net_billing(sim_df)
            deg_pln = round(float(sim_df["deg_cost_pln"].sum()), 2)
            imp_mwh = round(float(sim_df["e_import_mwh"].sum()), 3)
            exp_mwh = round(float(sim_df["e_export_mwh"].sum()), 3)
            curt_mwh = round(float((sim_df["p_curt_kw"] * 0.25 / 1000.0).sum()), 3)
            neg_exp_mwh = float(sim_df.loc[sim_df["rce_raw_pln_mwh"] <= 0.0, "e_export_mwh"].sum())
            if st == "uncontrolled":
                unc_neg_export = neg_exp_mwh

            prosumer_rows.append(
                {
                    "tariff": tf_spec.kind,
                    "strategy": st,
                    "import_mwh": imp_mwh,
                    "export_mwh": exp_mwh,
                    "curtailed_mwh": curt_mwh,
                    "neg_rce_export_mwh": round(neg_exp_mwh, 3),
                    "neg_rce_export_avoided_mwh": round(max(0.0, unc_neg_export - neg_exp_mwh), 3),
                    **bill,
                    "bess_deg_cost_pln": deg_pln,
                    "total_economic_cost_pln": round(bill["net_cash_bill_pln"] + deg_pln, 2),
                }
            )

    df_prosumer_summary = pd.DataFrame(prosumer_rows)
    df_prosumer_summary.to_csv(results_dir / "prosumer_summary.csv", index=False)

    # C&I B23 simulation
    df_ci_prices = build_price_frame(
        ts_utc=ts_utc,
        sdac_pln_mwh=df_in["sdac_pln_mwh"].to_numpy(),
        rce_pln_mwh=df_in["rce_pln_mwh"].to_numpy(),
        tariff=TariffSpec(name="C&I B23 Dynamic", kind="CI_B23_DYNAMIC"),
    )
    ci_results = simulate_ci_site(df_ci_prices, pv_unit_kw=pv_avail_kw)
    pd.DataFrame.from_dict(ci_results, orient="index").index.name = "mode"
    pd.DataFrame.from_dict(ci_results, orient="index").to_csv(results_dir / "ci_b23_summary.csv")

    # Closed-loop Modbus TCP HIL verification
    df_dyn_mpc = saved_runs[("DYNAMIC_G12w", "mpc_forecast")]
    df_hil = run_modbus_hil_verification(df_dyn_mpc)
    df_hil.to_csv(results_dir / "modbus_hil_trace.csv", index=False)

    # VPP Fleet Aggregation benchmark (250 HA Homes + 15 C&I B23 sites)
    vpp_summary = simulate_vpp_fleet_benchmark(df_dyn_mpc, n_homes=250, n_ci_sites=15)
    (results_dir / "vpp_fleet_summary.json").write_text(json.dumps(vpp_summary, indent=2), encoding="utf-8")

    # Build 24h (96-step) Live Home Assistant Entity Replay for 2026-05-01
    ts_l = df_dyn_mpc["ts_local"]
    mask_may1 = (ts_l >= pd.Timestamp("2026-05-01 00:00:00", tz="Europe/Warsaw")) & (
        ts_l < pd.Timestamp("2026-05-02 00:00:00", tz="Europe/Warsaw")
    )
    sub_may1 = df_dyn_mpc.loc[mask_may1]
    ha_24h_frames = []
    for _, r in sub_may1.iterrows():
        rce_v = float(r["rce_raw_pln_mwh"])
        buy_v = float(r["energy_buy_pln_mwh"] + r["dist_buy_pln_mwh"])
        pv_v = float(r["pv_avail_kw"])
        ld_v = float(r["base_load_kw"])
        sg = compute_sg_ready_state(rce_v, buy_v, max(0.0, pv_v - ld_v), 2.5)
        ev_a = compute_ocpp_ev_current_a(float(r["p_ev_kw"]))
        ha_24h_frames.append(
            {
                "time_str": r["ts_local"].strftime("%H:%M"),
                "rce": round(rce_v, 1),
                "buy": round(buy_v, 1),
                "pv_w": int(round(pv_v * 1000)),
                "bess_w": int(round((float(r["p_ch_kw"]) - float(r["p_dis_kw"])) * 1000)),
                "curt_w": int(round(float(r["p_curt_kw"]) * 1000)),
                "soc_pct": round(float(r["soc_kwh"]) * 10.0, 1),
                "zero_export": bool(rce_v <= 0.0),
                "sg_code": sg["state_code"],
                "sg_mode": sg["mode_num"],
                "sg_label": sg["label"],
                "ev_amps": ev_a,
            }
        )
    (results_dir / "ha_live_entities_24h.json").write_text(json.dumps(ha_24h_frames, indent=2), encoding="utf-8")

    inverter_profiles_payload = {
        k: {
            "display_name": v.display_name,
            "default_port": v.default_port,
            "unit_id": v.unit_id,
            "protocol_note": v.protocol_note,
            "registers": v.registers,
            "yaml_template": render_vendor_ha_yaml(k),
        }
        for k, v in INVERTER_PROFILES.items()
    }

    # Home Assistant artifacts
    (results_dir / "ha_config_example.yaml").write_text(render_ha_modbus_yaml(), encoding="utf-8")
    (results_dir / "mqtt_discovery.json").write_text(
        json.dumps(build_mqtt_discovery_payloads(), indent=2), encoding="utf-8"
    )
    _ = format_state_message(bess_target_w=3200, export_limit_w=0, rce_pln_mwh=-45.0, sdac_pln_mwh=-12.0, soc_pct=64.5)

    # 72-hour trace (1-3 May 2026) and Day-3 HIL trace for interactive Plotly charts
    df_dyn_unc = saved_runs[("DYNAMIC_G12w", "uncontrolled")]
    win_72 = (ts_l >= pd.Timestamp("2026-05-01 00:00:00", tz="Europe/Warsaw")) & (
        ts_l < pd.Timestamp("2026-05-04 00:00:00", tz="Europe/Warsaw")
    )
    if win_72.sum() < 96:
        win_72 = slice(0, 96 * 3)
    s_mpc72 = df_dyn_mpc.loc[win_72]
    s_unc72 = df_dyn_unc.loc[win_72]
    week_trace = {
        "ts": [t.strftime("%Y-%m-%d %H:%M") for t in s_mpc72["ts_local"]],
        "buy_price": [round(float(x), 1) for x in (s_mpc72["energy_buy_pln_mwh"] + s_mpc72["dist_buy_pln_mwh"])],
        "rce_price": [round(float(x), 1) for x in s_mpc72["rce_raw_pln_mwh"]],
        "pv_kw": [round(float(x), 2) for x in s_mpc72["pv_avail_kw"]],
        "batt_kw": [round(float(x), 2) for x in (s_mpc72["p_ch_kw"] - s_mpc72["p_dis_kw"])],
        "flex_kw": [round(float(x), 2) for x in (s_mpc72["p_hp_kw"] + s_mpc72["p_ev_kw"])],
        "curt_kw": [round(float(x), 2) for x in s_mpc72["p_curt_kw"]],
        "soc_unc": [round(float(x) * 10.0, 1) for x in s_unc72["soc_kwh"]],
        "soc_mpc": [round(float(x) * 10.0, 1) for x in s_mpc72["soc_kwh"]],
    }
    sub_hil3 = df_hil.iloc[192:288]
    hil_day3 = {
        "ts": [pd.Timestamp(t).strftime("%Y-%m-%d %H:%M") for t in sub_hil3["timestamp_local"]],
        "cmd_kw": [round(float(x) / 1000.0, 2) for x in sub_hil3["cmd_batt_w"]],
        "act_kw": [round(float(x) / 1000.0, 2) for x in sub_hil3["actual_batt_w"]],
        "curt_kw": [round(float(x) / 1000.0, 2) for x in sub_hil3["curtailed_pv_w"]],
        "soc": [round(float(x), 1) for x in sub_hil3["soc_pct"]],
        "wd": [int(x) for x in sub_hil3["watchdog_tripped"]],
        "comms_lost": [int(x) for x in sub_hil3["comms_lost"]],
    }

    # Summary JSON
    neg_rce_hours = float((df_in["rce_pln_mwh"] <= 0.0).sum() * 0.25)
    summary_payload = {
        "week_trace": week_trace,
        "hil_day3": hil_day3,
        "dataset_source": source_label,
        "period_utc": ["2025-10-01T00:00:00Z", "2026-09-30T23:45:00Z"],
        "n_quarters": int(len(df_in)),
        "pv_annual_generation_mwh": round(float((pv_avail_kw * 0.25 / 1000.0).sum()), 3),
        "negative_or_zero_rce_hours": round(neg_rce_hours, 2),
        "prosumer_summary": prosumer_rows,
        "ci_summary": ci_results,
        "vpp_summary": vpp_summary,
        "ha_24h_frames": ha_24h_frames,
        "inverter_profiles": inverter_profiles_payload,
        "modbus_hil": {
            "steps_verified": int(len(df_hil)),
            "Simulated_comms_drop_steps": int((df_hil["comms_lost"] > 0).sum()),
            "watchdog_tripped_steps": int((df_hil["watchdog_tripped"] > 0).sum()),
        },
    }
    (results_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    generate_plots(
        df_prosumer_summary=df_prosumer_summary,
        df_dyn_mpc=df_dyn_mpc,
        df_dyn_unc=saved_runs[("DYNAMIC_G12w", "uncontrolled")],
        ci_results=ci_results,
        df_hil=df_hil,
        vpp_summary=vpp_summary,
        ha_24h_frames=ha_24h_frames,
        img_dir=img_dir,
    )
    generate_dashboard_html(summary_payload, docs_dir=docs_dir)
    return summary_payload
