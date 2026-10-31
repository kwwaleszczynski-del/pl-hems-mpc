"""CLI entry point for `pl-hems-mpc`."""

from __future__ import annotations

from .pipeline import run_full_pipeline


def main() -> None:
    summary = run_full_pipeline()
    print(f"Completed pl-hems-mpc pipeline ({summary['dataset_source']}):")
    print(f"  Horizon: {summary['n_quarters']} x 15-min steps, PV yield = {summary['pv_annual_generation_mwh']} MWh/yr")
    print(f"  Hours with RCE <= 0 PLN/MWh: {summary['negative_or_zero_rce_hours']} h")
    dyn = [r for r in summary["prosumer_summary"] if r["tariff"] == "DYNAMIC_G12w"]
    for r in dyn:
        print(
            f"  [DYNAMIC_G12w | {r['strategy']:15s}] Net Cash Bill = {r['net_cash_bill_pln']:7.1f} PLN, "
            f"Wear = {r['bess_deg_cost_pln']:5.1f} PLN, Total = {r['total_economic_cost_pln']:7.1f} PLN"
        )
    print("  C&I C23 Summary:")
    for k, v in summary["ci_summary"].items():
        print(
            f"    [{k:18s}] Total Net = {v['annual_total_net_cost_pln']:8.0f} PLN/yr, "
            f"CapFee = {v['annual_capacity_fee_pln']:6.0f} PLN (A={v['effective_k_mult']:.2f}, K1={v['k1_days_share_pct']}%)"
        )


if __name__ == "__main__":
    main()
