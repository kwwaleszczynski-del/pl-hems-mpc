"""Polish retail & C&I electricity tariffs, net-billing deposit rules, and PSE capacity fee (K-factor).

Energy rates and the household (G) distribution rates in `TariffSpec` are illustrative round numbers, not a specific
supplier's or DSO's 2025/2026 tariff; the B23 variable charges come from one DSO's 2026 tariff. Statutory rules
carry a source in the docstrings; the legal state changes often, so every rule here is a simplification to be
verified against the current act before any real use.
"""

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
    # B23 three-zone variable charges (morning peak, evening peak, rest): TAURON Dystrybucja tariff for 2026 (URE
    # decision DRE.WRE.4211.54.8.2025.ASa1 of 17 Dec 2025), network rate 49.64 / 56.10 / 37.48 (table 7.1) + quality
    # rate 33.16 (from 1 Feb 2026) + OZE fee 7.30 + cogeneration fee 3.00, all PLN/MWh net. Applied to the whole
    # simulated window, including Oct-Dec 2025. To be verified.
    dist_b23_morning_pln_mwh: float = 93.10
    dist_b23_evening_pln_mwh: float = 99.56
    dist_b23_offpeak_pln_mwh: float = 80.94
    # Capacity fee (opłata mocowa) for non-household consumers, net PLN/MWh drawn in 07:00-21:59 on working days:
    # 0.1412 PLN/kWh for 1 Jul - 31 Dec 2025 (URE Informacja nr 26/2025 of 30 May 2025) and 0.2194 PLN/kWh for
    # 2026 (URE Informacja nr 58/2025 of 30 Oct 2025); hours per URE Informacja nr 57/2025. To be verified.
    capacity_fee_2025_pln_mwh: float = 141.20
    capacity_fee_2026_pln_mwh: float = 219.40


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
    - `cap_fee_base_pln_mwh`: capacity fee rate in its hours (C&I B23 only), before the K-factor,
    - `rce_raw_pln_mwh`: raw PSE RCE price,
    - `rce_credit_pln_mwh`: net-billing deposit credit per exported MWh in net-of-VAT terms, max(0, RCE).
      The statute credits 1.23 x the value to the deposit, which is then set against VAT-inclusive bills
      (amendment Dz. U. 2024 poz. 1847, as described by the Ministry of Climate in its 23 Jul 2025 answer to
      interpellation 10848); dividing both sides by 1.23 gives the net-of-VAT credit used here. To be verified.
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
        # Capacity fee hours for 2025 and 2026: 15 hours from 07:00 to 21:59 on working days
        # (URE Informacja nr 55/2024 and nr 57/2025)
        cap_window = (~free_day) & (hour >= 7) & (hour < 22)
    else:
        raise ValueError(f"Unknown tariff kind: {tariff.kind}")

    in_2026 = np.asarray(ts_local >= pd.Timestamp("2026-01-01", tz="Europe/Warsaw"))
    cap_rate = np.where(in_2026, tariff.capacity_fee_2026_pln_mwh, tariff.capacity_fee_2025_pln_mwh)
    cap_fee_base = np.where(cap_window, cap_rate, 0.0)
    # Negative RCE counts as zero for an individual prosumer (art. 4b ust. 2 pkt 2 of the RES Act,
    # Dz. U. 2024 poz. 1361; to be verified against later amendments)
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
    refund_cap_share: float = 0.30,
) -> dict[str, float]:
    """Settle a full year (or multi-year run) under a simplified Polish prosumer net-billing deposit.

    Rules as modelled (net of VAT, see `build_price_frame`); sources: RES Act art. 4b-4c and art. 4 ust. 11
    (Dz. U. 2024 poz. 1361) and the Ministry of Climate answer to interpellation 10848 (23 Jul 2025) on the
    amendment Dz. U. 2024 poz. 1847. The legal state changes often; to be verified before any real use.
    1. Exports of month `m` credited at `max(0, RCE_t)` form a deposit bucket `D_m`, assigned to the account
       in the following month and usable for 12 months from assignment (art. 4c ust. 5).
    2. Imported active energy cost (`E_imp * energy_buy_pln_mwh`) draws on the oldest bucket first (art. 4c
       ust. 7); anything not covered is paid in cash.
    3. Distribution and other network fees (`E_imp * dist_buy_pln_mwh`) are always paid in cash.
    4. A bucket unused after 12 months is refunded up to `refund_cap_share` of `D_m` and the rest expires.
       The default 30 % is the cap the Ministry gives for prosumers settled at RCE (20 % for RCEm).
    5. Buckets still open at the end of the simulated horizon are settled with the same cap (conservative:
       in reality they could still cover imports after the horizon).
    """
    ts_local = df_sim.index.tz_convert("Europe/Warsaw")
    ym = ts_local.year * 100 + ts_local.month
    unique_months = np.unique(ym)

    # FIFO queue of [assigned_month_idx, initial_credit_pln, remaining_credit_pln]
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

        # Deduct active energy cost FIFO from buckets already assigned (earlier months' exports)
        to_cover = act_cost
        for b in buckets:
            if to_cover <= 0:
                break
            used = min(b[2], to_cover)
            b[2] -= used
            to_cover -= used
            total_deposit_used += used

        total_cash_active_energy += to_cover

        # This month's credit is assigned to the deposit next month
        if exp_credit > 0:
            buckets.append([float(m_idx + 1), exp_credit, exp_credit])

    # Settle any remaining deposit at the end of the horizon with the same refund cap (30 % by default)
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
        "deposit_forfeited_pln": round(total_expired_forfeited + forfeited_end, 2),
        "net_cash_bill_pln": round(net_annual_bill_pln, 2),
    }


# Art. 70a ust. 1 and 5 of the Capacity Market Act (ustawa o rynku mocy, t.j. Dz. U. 2023 poz. 2131): upper bound
# of the peak/off-peak difference [%] and coefficient A for groups K1..K4. To be verified against later amendments.
K_BANDS: list[tuple[float, float, str]] = [(5.0, 0.17, "K1"), (10.0, 0.50, "K2"), (15.0, 0.83, "K3"), (np.inf, 1.00, "K4")]


def k_band(delta_pct: float) -> tuple[float, str]:
    """Return (coefficient A, label) for a peak/off-peak difference in percent."""
    for upper, a_coef, label in K_BANDS:
        if delta_pct < upper:
            return a_coef, label
    return K_BANDS[-1][1], K_BANDS[-1][2]


def compute_ci_k_factor(
    e_import_mwh: np.ndarray,
    cap_window: np.ndarray,
    ts_utc: pd.DatetimeIndex,
    rate_pln_mwh: float | np.ndarray,
) -> dict[str, float]:
    """Compute the daily capacity-fee group (K1..K4) and the capacity fee of a non-household site.

    Art. 70a of the Capacity Market Act (t.j. Dz. U. 2023 poz. 2131), as read here: the qualification period is
    one day (ust. 3); on a working day the difference is mean consumption in the fee hours over mean consumption
    in the other hours, minus one (ust. 2); groups K1..K4 and coefficients A are in `K_BANDS`, with K4 also when
    off-hour consumption is zero (ust. 1); the fee is A x energy drawn in the fee hours x rate (ust. 4).
    Means are taken per quarter-hour, so 23- and 25-hour days are handled. To be verified before real use.
    """
    ts_local = ts_utc.tz_convert("Europe/Warsaw")
    rate = np.broadcast_to(np.asarray(rate_pln_mwh, dtype=float), e_import_mwh.shape)
    df = pd.DataFrame(
        {
            "date": ts_local.date,
            "free": is_polish_holiday_or_weekend(ts_local),
            "peak": cap_window,
            "e_mwh": e_import_mwh,
            "fee_base_pln": e_import_mwh * rate,
        }
    )
    work = df[~df["free"]]
    pk = work[work["peak"]].groupby("date").agg(e=("e_mwh", "sum"), n=("e_mwh", "size"), fee=("fee_base_pln", "sum"))
    off = work[~work["peak"]].groupby("date").agg(e=("e_mwh", "sum"), n=("e_mwh", "size"))

    total_peak_mwh = float(pk["e"].sum())
    total_off_mwh = float(off["e"].sum())
    mean_p_peak = total_peak_mwh / max(1, int(pk["n"].sum()))
    mean_p_off = total_off_mwh / max(1, int(off["n"].sum()))
    if mean_p_off <= 1e-12:
        delta_pct = 100.0 if mean_p_peak > 0 else 0.0
    else:
        delta_pct = max(0.0, (mean_p_peak / mean_p_off - 1.0) * 100.0)

    k_counts = {label: 0 for _, _, label in K_BANDS}
    total_fee_daily_pln = 0.0
    for d, row in pk.iterrows():
        m_pk = row["e"] / row["n"]
        m_of = off["e"].get(d, 0.0) / max(1, off["n"].get(d, 0))
        if m_of <= 1e-12:
            d_pct = 0.0 if m_pk <= 1e-12 else np.inf
        else:
            d_pct = max(0.0, (m_pk / m_of - 1.0) * 100.0)
        a_coef, label = k_band(d_pct)
        k_counts[label] += 1
        total_fee_daily_pln += float(row["fee"]) * a_coef

    unmodulated_fee_pln = float(pk["fee"].sum())
    effective_a = total_fee_daily_pln / unmodulated_fee_pln if unmodulated_fee_pln > 0 else 0.0
    n_days = max(1, sum(k_counts.values()))

    return {
        "peak_import_mwh": round(total_peak_mwh, 3),
        "offpeak_import_mwh": round(total_off_mwh, 3),
        "mean_delta_pct": round(delta_pct, 2),
        "effective_k_multiplier": round(effective_a, 3),
        "capacity_fee_unmodulated_pln": round(unmodulated_fee_pln, 2),
        "capacity_fee_actual_pln": round(total_fee_daily_pln, 2),
        "k1_days_share_pct": round(100.0 * k_counts["K1"] / n_days, 1),
        "k_days": k_counts,
    }
