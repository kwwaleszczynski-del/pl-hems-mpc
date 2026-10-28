"""Polish retail & C&I electricity tariffs, net-billing deposit rules, and PSE capacity fee (K-factor)."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import numpy as np
import pandas as pd


# Fixed Polish statutory holidays (month, day) + Easter-dependent holidays for 2024-2026
FIXED_HOLIDAYS = {
    (1, 1),   # Nowy Rok
    (1, 6),   # Trzech Króli
    (5, 1),   # Święto Pracy
    (5, 3),   # Święto Konstytucji 3 Maja
    (8, 15),  # Wniebowzięcie NMP
    (11, 1),  # Wszystkich Świętych
    (11, 11), # Narodowe Święto Niepodległości
    (12, 24), # Wigilia (statutory from 2025)
    (12, 25), # Boże Narodzenie (1)
    (12, 26), # Boże Narodzenie (2)
}

MOVABLE_HOLIDAYS = {
    dt.date(2024, 3, 31), dt.date(2024, 4, 1), dt.date(2024, 5, 19), dt.date(2024, 5, 30),
    dt.date(2025, 4, 20), dt.date(2025, 4, 21), dt.date(2025, 6, 8), dt.date(2025, 6, 19),
    dt.date(2026, 4, 5),  dt.date(2026, 4, 6),  dt.date(2026, 5, 24), dt.date(2026, 6, 4),
}


def is_polish_holiday_or_weekend(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Return boolean mask for Saturdays, Sundays, and Polish statutory holidays."""
    if isinstance(dates, pd.DatetimeIndex):
        d_date = dates.date
        weekday = dates.day_of_week
        month = dates.month
        day = dates.day
    else:
        dt_acc = dates.dt
        d_date = dt_acc.date
        weekday = dt_acc.day_of_week
        month = dt_acc.month
        day = dt_acc.day

    is_wknd = np.asarray(weekday >= 5, dtype=bool)
    md = list(zip(np.asarray(month), np.asarray(day)))
    is_fixed = np.array([m in FIXED_HOLIDAYS for m in md], dtype=bool)
    is_mov = np.array([d in MOVABLE_HOLIDAYS for d in d_date], dtype=bool)
    return is_wknd | is_fixed | is_mov


@dataclass(frozen=True)
class TariffSpec:
    """Parameters of a Polish household (G) or commercial/industrial (C/B) tariff (PLN/MWh net)."""
    name: str
    kind: str  # "G11", "G12w", "DYNAMIC_G12w", "CI_B23_DYNAMIC"
    # Active energy prices for flat / 2-zone tariffs (PLN/MWh)
    energy_flat_pln_mwh: float = 620.0
    energy_peak_pln_mwh: float = 720.0
    energy_offpeak_pln_mwh: float = 430.0
    # Dynamic tariff supplier margin + trading fee over SDAC (PLN/MWh)
    dynamic_margin_pln_mwh: float = 85.0
    # Variable distribution rates (składnik zmienny sieciowy + jakościowy + OZE + kogeneracyjna, PLN/MWh)
    dist_flat_pln_mwh: float = 295.0
    dist_peak_pln_mwh: float = 345.0
    dist_offpeak_pln_mwh: float = 95.0
    # B23 three-zone distribution (morning peak, evening peak, rest)
    dist_b23_morning_pln_mwh: float = 240.0
    dist_b23_evening_pln_mwh: float = 330.0
    dist_b23_offpeak_pln_mwh: float = 78.0
    # PSE Capacity Fee (opłata mocowa, PLN/MWh in peak hours 07:00-21:59 on working days)
    capacity_fee_base_pln_mwh: float = 141.20


def build_price_frame(
    ts_utc: pd.DatetimeIndex,
    sdac_pln_mwh: np.ndarray,
    rce_pln_mwh: np.ndarray,
    tariff: TariffSpec,
) -> pd.DataFrame:
    """Build 15-minute price series in local time (Europe/Warsaw) for a given tariff.

    Separates:
    - `energy_buy_pln_mwh`: active energy component (can be offset by the prosumer net-billing deposit),
    - `dist_buy_pln_mwh`: variable distribution + quality/RES/cogeneration fees (always paid in cash),
    - `capacity_fee_pln_mwh`: base PSE capacity fee in peak hours (for C&I B23),
    - `rce_raw_pln_mwh`: raw PSE RCE price,
    - `rce_credit_pln_mwh`: net-billing deposit credit per exported MWh, equal to max(0, RCE) * 1.23
      (or net equivalent max(0, RCE) when both sides are compared net of VAT).
    """
    ts_local = ts_utc.tz_convert("Europe/Warsaw")
    hour = np.asarray(ts_local.hour)
    month = np.asarray(ts_local.month)
    free_day = is_polish_holiday_or_weekend(ts_local)

    # G12w off-peak windows: 22:00-05:59, 13:00-14:59 on working days, plus all weekend/holiday hours
    g12w_offpeak = free_day | (hour >= 22) | (hour < 6) | ((hour >= 13) & (hour < 15))

    if tariff.kind == "G11":
        energy_buy = np.full(len(ts_utc), tariff.energy_flat_pln_mwh, dtype=float)
        dist_buy = np.full(len(ts_utc), tariff.dist_flat_pln_mwh, dtype=float)
        cap_window = np.zeros(len(ts_utc), dtype=bool)
    elif tariff.kind == "G12w":
        energy_buy = np.where(g12w_offpeak, tariff.energy_offpeak_pln_mwh, tariff.energy_peak_pln_mwh)
        dist_buy = np.where(g12w_offpeak, tariff.dist_offpeak_pln_mwh, tariff.dist_peak_pln_mwh)
        cap_window = np.zeros(len(ts_utc), dtype=bool)
    elif tariff.kind == "DYNAMIC_G12w":
        energy_buy = np.maximum(0.0, sdac_pln_mwh + tariff.dynamic_margin_pln_mwh)
        dist_buy = np.where(g12w_offpeak, tariff.dist_offpeak_pln_mwh, tariff.dist_peak_pln_mwh)
        cap_window = np.zeros(len(ts_utc), dtype=bool)
    elif tariff.kind == "CI_B23_DYNAMIC":
        energy_buy = sdac_pln_mwh + tariff.dynamic_margin_pln_mwh
        # B23 zones: morning peak 07:00-12:59, evening peak Oct-Mar 16:00-20:59 / Apr-Sep 19:00-21:59
        is_winter = (month >= 10) | (month <= 3)
        morn_peak = (~free_day) & (hour >= 7) & (hour < 13)
        eve_peak = (~free_day) & (
            (is_winter & (hour >= 16) & (hour < 21))
            | ((~is_winter) & (hour >= 19) & (hour < 22))
        )
        dist_buy = np.where(
            eve_peak,
            tariff.dist_b23_evening_pln_mwh,
            np.where(morn_peak, tariff.dist_b23_morning_pln_mwh, tariff.dist_b23_offpeak_pln_mwh),
        )
        # Statutory PSE capacity fee peak window: working days 07:00-21:59 (15 hours/day)
        cap_window = (~free_day) & (hour >= 7) & (hour < 22)
    else:
        raise ValueError(f"Unknown tariff kind: {tariff.kind}")

    cap_fee_base = np.where(cap_window, tariff.capacity_fee_base_pln_mwh, 0.0)
    # Under Polish RES Act art. 4b, negative RCE prices credit 0 PLN to the prosumer deposit
    rce_credit = np.maximum(0.0, rce_pln_mwh)

    return pd.DataFrame(
        {
            "ts_local": ts_local,
            "sdac_pln_mwh": sdac_pln_mwh,
            "rce_raw_pln_mwh": rce_pln_mwh,
            "rce_credit_pln_mwh": rce_credit,
            "energy_buy_pln_mwh": energy_buy,
            "dist_buy_pln_mwh": dist_buy,
            "total_buy_no_cap_pln_mwh": energy_buy + dist_buy,
            "cap_window": cap_window,
            "cap_fee_base_pln_mwh": cap_fee_base,
            "is_offpeak": g12w_offpeak,
        },
        index=ts_utc,
    )


def settle_prosumer_net_billing(
    df_sim: pd.DataFrame,
    refund_cap_share: float = 0.20,
) -> dict[str, float]:
    """Settle a full year (or multi-year run) under the Polish prosumer net-billing deposit rules.

    Rules (art. 4 ust. 1-12 ustawy o OZE, simplified to net PLN):
    1. Each month `m`, exported energy credited at `max(0, RCE_t)` adds `D_m` PLN to the prosumer's
       12-month FIFO deposit bucket.
    2. Imported active energy cost (`E_imp * energy_buy_pln_mwh`) is deducted first from the oldest
       available deposit bucket (up to 12 months old); any uncovered active energy is paid in cash.
    3. Distribution & quality/RES/cogeneration fees (`E_imp * dist_buy_pln_mwh`) cannot be paid from
       the deposit and are always paid in cash.
    4. Any deposit from month `m` still unused after 12 billing months expires, with at most `20%`
       of the month's original credit refunded in cash (`nadpłata <= 20%`).
    """
    ts_local = df_sim.index.tz_convert("Europe/Warsaw")
    ym = ts_local.year * 100 + ts_local.month
    unique_months = np.unique(ym)

    # FIFO queue of [month_idx, initial_credit_pln, remaining_credit_pln]
    buckets: list[list[float]] = []

    total_active_energy_cost = 0.0
    total_dist_cash = 0.0
    total_export_credit = 0.0
    total_deposit_used = 0.0
    total_cash_active_energy = 0.0
    total_cash_refund = 0.0
    total_expired_forfeited = 0.0

    for m_idx, m_code in enumerate(unique_months):
        mask = ym == m_code
        sub = df_sim.loc[mask]

        # Expire buckets older than 12 months
        still_valid: list[list[float]] = []
        for b_idx, init_cr, rem_cr in buckets:
            if m_idx - int(b_idx) >= 12:
                refundable = min(rem_cr, refund_cap_share * init_cr)
                total_cash_refund += refundable
                total_expired_forfeited += max(0.0, rem_cr - refundable)
            else:
                still_valid.append([b_idx, init_cr, rem_cr])
        buckets = still_valid

        # Month's export credit and import obligations
        exp_credit = float((sub["e_export_mwh"] * sub["rce_credit_pln_mwh"]).sum())
        act_cost = float((sub["e_import_mwh"] * sub["energy_buy_pln_mwh"]).sum())
        dist_cost = float((sub["e_import_mwh"] * sub["dist_buy_pln_mwh"]).sum())

        total_export_credit += exp_credit
        total_active_energy_cost += act_cost
        total_dist_cash += dist_cost

        if exp_credit > 0:
            buckets.append([float(m_idx), exp_credit, exp_credit])

        # Deduct active energy cost FIFO from buckets
        to_cover = act_cost
        for b in buckets:
            if to_cover <= 0:
                break
            used = min(b[2], to_cover)
            b[2] -= used
            to_cover -= used
            total_deposit_used += used

        total_cash_active_energy += to_cover

    # Settle any remaining deposit at the end of the horizon with the same 20% refund cap
    remaining_end = sum(b[2] for b in buckets)
    refundable_end = sum(min(b[2], refund_cap_share * b[1]) for b in buckets)
    forfeited_end = max(0.0, remaining_end - refundable_end)

    net_annual_bill_pln = (
        total_cash_active_energy
        + total_dist_cash
        - total_cash_refund
        - refundable_end
    )

    return {
        "active_energy_gross_pln": round(total_active_energy_cost, 2),
        "distribution_cash_pln": round(total_dist_cash, 2),
        "export_credit_gross_pln": round(total_export_credit, 2),
        "deposit_used_pln": round(total_deposit_used, 2),
        "active_energy_cash_pln": round(total_cash_active_energy, 2),
        "deposit_refund_cash_pln": round(total_cash_refund + refundable_end, 2),
        "deposit_forfeited_80pct_pln": round(total_expired_forfeited + forfeited_end, 2),
        "net_cash_bill_pln": round(net_annual_bill_pln, 2),
    }


def compute_ci_k_factor(
    e_import_mwh: np.ndarray,
    cap_window: np.ndarray,
    ts_utc: pd.DatetimeIndex,
    base_rate_pln_mwh: float = 141.20,
) -> dict[str, float]:
    """Compute the Polish Capacity Market K-coefficient (K1..K4) and capacity fee for a C&I site.

    Under art. 70a of the Polish Capacity Market Act (ustawa o rynku mocy), on each working day
    the ratio delta_S of mean power in peak hours (07:00-21:59, 15 h) to mean power in off-peak
    hours (22:00-06:59, 9 h) determines the daily/monthly multiplier A (K1..K4):
      - delta < 5%   (ratio < 1.05): K1 -> A = 0.17 (83% discount!)
      - 5% <= delta < 10%          : K2 -> A = 0.50
      - 10% <= delta < 15%         : K3 -> A = 0.83
      - delta >= 15%               : K4 -> A = 1.00
    """
    ts_local = ts_utc.tz_convert("Europe/Warsaw")
    dates = ts_local.date
    free_day = is_polish_holiday_or_weekend(ts_local)

    df = pd.DataFrame(
        {"date": dates, "free": free_day, "peak": cap_window, "e_mwh": e_import_mwh}
    )
    work = df[~df["free"]]

    daily_peak = work[work["peak"]].groupby("date")["e_mwh"].sum() / 15.0
    daily_off = work[~work["peak"]].groupby("date")["e_mwh"].sum() / 9.0

    # Aggregate over billing months (as OSD settles monthly/daily for C&I)
    total_peak_mwh = float(work[work["peak"]]["e_mwh"].sum())
    total_off_mwh = float(work[~work["peak"]]["e_mwh"].sum())

    mean_p_peak = total_peak_mwh / max(1.0, len(daily_peak) * 15.0)
    mean_p_off = total_off_mwh / max(1.0, len(daily_off) * 9.0)

    if mean_p_off <= 1e-9:
        delta_pct = 100.0 if mean_p_peak > 0 else 0.0
    else:
        delta_pct = max(0.0, (mean_p_peak / mean_p_off - 1.0) * 100.0)

    # Also compute daily K-class distribution
    k_counts = {"K1": 0, "K2": 0, "K3": 0, "K4": 0}
    total_fee_daily_pln = 0.0
    for d in daily_peak.index:
        p_pk = float(daily_peak.loc[d])
        p_of = float(daily_off.get(d, 0.0))
        d_pct = 0.0 if p_pk <= 1e-9 else (100.0 if p_of <= 1e-9 else max(0.0, (p_pk / p_of - 1.0) * 100.0))
        if d_pct < 5.0:
            k_mult, k_lbl = 0.17, "K1"
        elif d_pct < 10.0:
            k_mult, k_lbl = 0.50, "K2"
        elif d_pct < 15.0:
            k_mult, k_lbl = 0.83, "K3"
        else:
            k_mult, k_lbl = 1.00, "K4"
        k_counts[k_lbl] += 1
        total_fee_daily_pln += (p_pk * 15.0) * base_rate_pln_mwh * k_mult

    unmodulated_fee_pln = total_peak_mwh * base_rate_pln_mwh
    effective_a = total_fee_daily_pln / unmodulated_fee_pln if unmodulated_fee_pln > 0 else 0.17

    return {
        "peak_import_mwh": round(total_peak_mwh, 3),
        "offpeak_import_mwh": round(total_off_mwh, 3),
        "mean_delta_pct": round(delta_pct, 2),
        "effective_k_multiplier": round(effective_a, 3),
        "capacity_fee_unmodulated_pln": round(unmodulated_fee_pln, 2),
        "capacity_fee_actual_pln": round(total_fee_daily_pln, 2),
        "k1_days_share_pct": round(100.0 * k_counts["K1"] / max(1, sum(k_counts.values())), 1),
    }
