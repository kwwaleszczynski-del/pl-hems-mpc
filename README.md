# pl-hems-mpc: 15-Minute MILP Home & C&I Energy Management System with Modbus TCP (SunSpec) Hardware-in-the-Loop

[![CI](https://github.com/kwwaleszczynski-del/pl-hems-mpc/actions/workflows/ci.yml/badge.svg)](https://github.com/kwwaleszczynski-del/pl-hems-mpc/actions/workflows/ci.yml)
[![Live Dashboard](https://img.shields.io/badge/GitHub%20Pages-Live%20Interactive%20EMS%20Calculator-10b981)](https://kwwaleszczynski-del.github.io/pl-hems-mpc/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-38bdf8.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-amber.svg)](LICENSE)

> **Interactive Web Calculator & Home Assistant Configurator:** [**kwwaleszczynski-del.github.io/pl-hems-mpc/**](https://kwwaleszczynski-del.github.io/pl-hems-mpc/)

An engineering-grade **Model Predictive Control (MPC / MILP)** energy management engine and **closed-loop Modbus TCP (SunSpec) Hardware-in-the-Loop (HIL)** testbed tailored to the **Polish 15-minute electricity market** (`2025-10-01` to `2026-09-30`, `N = 35,040` quarter-hour steps).

It bridges the gap between **wholesale/retail tariff mathematics** (Polish prosumer `net-billing` under *Art. 4 ustawy o OZE*, `15-min PSE RCE`, `TGE SDAC`, and C&I `Art. 70a` Capacity Market `K1..K4` modulation) and **field-level hardware execution** (Home Assistant MQTT Auto-Discovery + binary Modbus TCP `FC 0x03 / 0x06 / 0x10` register control with a hardware watchdog safety fallback).

---

## Key Quantitative Findings (`2025-10-01` to `2026-09-30`, Real PSE RCE + TGE SDAC + ERA5)

Across the first full 12-month billing year of 15-minute `SDAC` and `RCE` settlement in Poland (`35,040` quarter-hours), **PSE RCE was zero or negative for `357.5 hours` (`1,430` fifteen-minute intervals, reaching `-1,457.59 PLN/MWh` on `2026-05-01`)**.

### 1. Residential Prosumer (`10 kWp PV + 10 kWh / 5 kW LFP BESS + 2R2C Floor-Slab Heat Pump + 11 kW EV`)

| Retail Tariff | Control Strategy | Grid Import [MWh/yr] | Grid Export [MWh/yr] | PV Curtailed [MWh/yr] | Neg-RCE Export [MWh/yr] | Net Cash Bill [PLN/yr] | BESS Wear [PLN/yr] | **Total Economic Cost [PLN/yr]** | **Savings vs. Uncontrolled** |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **G11 Flat** | `1. Uncontrolled` | 8.279 | 6.594 | 0.000 | 1.673 | 6,326 | 373 | **6,700 PLN** | — |
| **G11 Flat** | `2. Neg-RCE Guard` | 8.284 | 4.986 | 1.614 | 0.000 | 6,293 | 373 | **6,666 PLN** | `-34 PLN` (`-0.5%`) |
| **G11 Flat** | `3. HA YAML Heuristic` | 8.114 | 4.761 | 1.562 | 0.000 | 6,212 | 369 | **6,581 PLN** | `-119 PLN` (`-1.8%`) |
| **G11 Flat** | **`4. 15-min MILP-MPC (Forecast)`** | **7.368** | **4.796** | **0.988** | **0.000** | **5,551** | **330** | **5,881 PLN** | **`-819 PLN` (`-12.2%`)** |
| **G11 Flat** | `5. 15-min MILP (Oracle Bound)` | 7.306 | 4.819 | 0.898 | 0.000 | 5,429 | 334 | **5,763 PLN** | `-937 PLN` (`-14.0%`) |
| **G12w Two-Zone** | `1. Uncontrolled` | 8.279 | 6.594 | 0.000 | 1.673 | 5,189 | 373 | **5,563 PLN** | — |
| **G12w Two-Zone** | `2. Neg-RCE Guard` | 8.284 | 4.986 | 1.614 | 0.000 | 4,098 | 373 | **4,471 PLN** | `-1,092 PLN` (`-19.6%`) |
| **G12w Two-Zone** | `3. HA YAML Heuristic` | 8.328 | 4.834 | 1.580 | 0.000 | 3,967 | 369 | **4,336 PLN** | `-1,227 PLN` (`-22.0%`) |
| **G12w Two-Zone** | **`4. 15-min MILP-MPC (Forecast)`** | **8.381** | **5.577** | **1.108** | **0.000** | **3,150** | **425** | **3,574 PLN** | **`-1,988 PLN` (`-35.7%`)** |
| **G12w Two-Zone** | `5. 15-min MILP (Oracle Bound)` | 8.783 | 6.084 | 0.971 | 0.000 | 2,970 | 478 | **3,448 PLN** | `-2,115 PLN` (`-38.0%`) |
| **Dynamic SDAC+G12w** | `1. Uncontrolled` | 8.279 | 6.594 | 0.000 | 1.673 | 5,904 | 373 | **6,277 PLN** | — |
| **Dynamic SDAC+G12w** | `2. Neg-RCE Guard` | 8.284 | 4.986 | 1.614 | 0.000 | 4,909 | 373 | **5,282 PLN** | `-995 PLN` (`-15.9%`) |
| **Dynamic SDAC+G12w** | `3. HA YAML Heuristic` | 8.283 | 4.817 | 1.593 | 0.000 | 4,591 | 394 | **4,985 PLN** | `-1,292 PLN` (`-20.6%`) |
| **Dynamic SDAC+G12w** | **`4. 15-min MILP-MPC (Forecast)`** | **7.716** | **4.973** | **1.023** | **0.000** | **3,573** | **427** | **4,001 PLN** | **`-2,277 PLN` (`-36.3%`)** |
| **Dynamic SDAC+G12w** | `5. 15-min MILP (Oracle Bound)` | 7.649 | 5.014 | 0.897 | 0.000 | 3,440 | 453 | **3,893 PLN** | `-2,384 PLN` (`-38.0%`) |

![Annual Prosumer Bill Comparison](docs/img/bill_comparison.png)

#### Structural Insights for Polish Prosumers:
1. **Why Uncontrolled Dynamic Tariffs Backfire (`+714 PLN/yr` penalty vs. `G12w`):** A household with an unscheduled EV (`17:30` plug-in) and standard heat-pump hysteresis switching from `G12w` (`5,563 PLN/yr`) to a `Dynamic SDAC` contract (`6,277 PLN/yr`) pays **more** because its evening load coincides with `1,000–1,400 PLN/MWh` SDAC peaks while its LFP battery fills up at `09:30` and leaves `1.673 MWh/yr` of midday PV exposed to `RCE <= 0 PLN/MWh` (which credits `0 PLN` to the prosumer deposit).
2. **MPC Beats Static Home Assistant Automations by `762–984 PLN/yr`:** Static YAML time-window rules (`3. ha_heuristic`) cannot adapt to day-specific solar curves or shifting SDAC/RCE spreads. Rolling 15-minute MILP-MPC (`4. mpc_forecast`) cuts total annual cost by **`2,277 PLN/yr` (`-36.3%`)** on `Dynamic SDAC+G12w` and **`1,988 PLN/yr` (`-35.7%`)** on `G12w`, capturing **`95.5%`** of the perfect-foresight Oracle bound despite realistic `12%` PV and `14%` load forecast errors.
3. **The `G12w` vs. `Dynamic SDAC` Asymmetry in Polish Net-Billing:** Under *Art. 4b ustawy o OZE*, prosumers on a two-zone `G12w` purchase contract still receive **dynamic 15-minute `RCE` credits** on exports during early-morning/late-afternoon solar ramps while remaining insured against winter dunkelflaute `SDAC` spikes (`720 PLN/MWh` peak / `430 PLN/MWh` off-peak active energy cap). As a result, `G12w + MILP-MPC` (`3,574 PLN/yr`) outperforms `Dynamic SDAC + MILP-MPC` (`4,001 PLN/yr`) by `427 PLN/yr`.

![72-Hour Dispatch Replay](docs/img/dispatch_week.png)

---

### 2. Commercial & Industrial (C&I) Site (`50 kWp PV + 100 kWh / 50 kW BESS` on `B23 Dynamic` + PSE Capacity Fee)

Under **Art. 70a of the Polish Capacity Market Act (*ustawa o rynku mocy*)**, C&I sites settled on tariff groups `B` and `C` pay a statutory **Capacity Fee (*opłata mocowa*, `141.20 PLN/MWh`)** on grid imports during working-day peak hours (`07:00–21:59`, 15 hours/day), multiplied by a daily/monthly **K-factor (`A in {0.17, 0.50, 0.83, 1.00}`)** determined by the ratio $\Delta_S$ of mean peak import power (`07:00–21:59`) to mean off-peak import power (`22:00–06:59`):
- **Band K1 ($\Delta_S < 5\%$):** $A = 0.17$ (**83% statutory discount** on the capacity fee!)
- **Band K2 ($5\% \le \Delta_S < 10\%$):** $A = 0.50$
- **Band K3 ($10\% \le \Delta_S < 15\%$):** $A = 0.83$
- **Band K4 ($\Delta_S \ge 15\%$):** $A = 1.00$

| C&I Operating Mode (`40 kW` Contracted Limit) | Energy + B23 Dist. [PLN/yr] | PSE Capacity Fee [PLN/yr] | Effective Multiplier $A$ | Working Days in **K1 Band** | Exceedance Penalty [PLN/yr] | BESS Wear [PLN/yr] | **Total Net Annual Cost [PLN/yr]** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1. 50 kWp PV Only (No BESS)` | 130,394 | 10,820 | `0.85` | `21.4%` | 5,099 | 0 | **146,046 PLN** |
| `2. PV + 100 kWh BESS (Factory Self-Cons.)` | 127,675 | 10,819 | `0.85` | `21.4%` | 5,099 | 545 | **144,111 PLN** |
| **`3. PV + 100 kWh BESS (15-min MILP-MPC)`** | **127,086** | **3,127** | **`0.30`** | **`87.3%`** | **3,697** | **2,459** | **136,105 PLN** |

![C&I Capacity Fee K-Factor Shaping](docs/img/ci_k_factor.png)

Factory self-consumption (`mode 2`) saves almost nothing on the PSE Capacity Fee (`10,819` vs. `10,820 PLN/yr`) because the battery discharges early in the afternoon and leaves evening working-day hours unbuffered. By shaping the peak/off-peak import ratio toward $\Delta_S < 5\%$, **15-minute MILP-MPC (`mode 3`) pushes `87.3%` of working days into Band K1 (`A = 0.17`), slashing the annual PSE Capacity Fee by `71.1%` (`-7,693 PLN/yr`) and reducing total net site cost by `9,941 PLN/yr`**.

---

## 3. Closed-Loop Modbus TCP Hardware-in-the-Loop (HIL) & Hardware Watchdog

To prove that the optimization schedules can be safely executed on real residential/C&I hybrid inverters (Sungrow `SH10RT`, Huawei `SUN2000`, GoodWe `ET`, Victron `ESS`, Deye `SUN-10K`), `src/hems_mpc/modbus_hil.py` implements a **pure-Python binary Modbus TCP server and client** (`7-byte MBAP header + PDU` for Function Codes `0x03`, `0x06`, `0x10`) coupled to a deterministic LFP + PCC physics engine.

### Emulated Hybrid Inverter Register Map (Unit ID `1`)

| Register | Type | R/W | Description | Fail-Safe / Watchdog Behavior |
| :--- | :--- | :---: | :--- | :--- |
| `40001` | `uint16` | R/W | **EMS Control Mode** (`0` = Local Self-Consumption, `2` = Forced MPC Dispatch) | Reverts to `0` if heartbeat stalls `> Reg 40005` |
| `40002` | `int16` | R/W | **Forced Battery Power Command [W]** (`+` Charge, `-` Discharge, two's complement) | Reverts to `0 W` on watchdog trip |
| `40003` | `uint16` | R/W | **Dynamic Grid Export Limit [W]** (`0 W` = Zero-Export during `RCE <= 0`) | Reverts to `10,000 W` on watchdog trip |
| `40004` | `uint16` | R/W | **EMS Watchdog Heartbeat Counter** (incremented every 15-min cycle) | Monitored by inverter state machine |
| `40005` | `uint16` | R/W | **Watchdog Timeout [s]** (default `900 s` = 15 min) | Configurable timeout threshold |
| `40010` | `uint16` | R | **Battery State of Charge [0.1 %]** (`100..950` -> `10.0%..95.0%`) | Enforced by inverter BMS limits |
| `40011` | `uint16` | R | **Available Rooftop PV AC Power [W]** | Telemetry |
| `40012` | `uint16` | R | **Total Site Load Power [W]** | Telemetry |
| `40013` | `int16` | R | **Actual Executed Battery Power [W]** (`+` Charging, `-` Discharging) | Clamped by BMS SoC & power ratings |
| `40014` | `int16` | R | **Grid PCC Active Power [W]** (`+` Grid Import, `-` Grid Export) | Verified against PCC limits |
| `40015` | `uint16` | R | **Curtailed PV Power [W]** | Active when `Export > Reg 40003` |
| `40016` | `uint16` | R | **Watchdog Tripped Status Flag** (`0` = Normal, `1` = Fallback Active) | Cleared when `Reg 40004` resumes |

![Modbus TCP HIL Loop Verification](docs/img/modbus_hil_loop.png)

In our 7-day closed-loop HIL run (`672` quarter-hour TCP socket transactions over `127.0.0.1`), we inject a **45-minute Home Assistant / Wi-Fi outage on `2026-05-01 12:00–12:45`** right during the `-1,458 PLN/MWh` negative-RCE solar peak:
- At `12:15` (`t = +900 s` without a heartbeat increment on `Reg 40004`), the virtual inverter trips `Reg 40016 = 1` and safely falls back to `Reg 40001 = 0` (`Local Self-Consumption`).
- At `12:45`, when the EMS client resumes `FC 0x10` writes, `Reg 40016` clears to `0` and zero-export curtailment (`Reg 40003 = 0 W`) immediately resumes.

---

## 4. Home Assistant HACS Integration, Multi-Brand Inverter Profiles & VPP Fleet Aggregator

### A. Installable Home Assistant Custom Component (`custom_components/pl_hems_mpc/`)
The repository includes a HACS-compatible Home Assistant integration (`hacs.json` + `custom_components/pl_hems_mpc/`) that exposes 8 live entities for Polish prosumers and C&I sites:
- `sensor.pl_hems_pse_rce_15min` — current 15-min PSE RCE price (`PLN/MWh`) and net-billing credit (`max(0, RCE)`).
- `sensor.pl_hems_fifo_deposit_balance` — live 12-month FIFO prosumer deposit tracker with `at_risk_80pct_forfeiture_pln`.
- `sensor.pl_hems_ci_k_factor_live` — real-time C&I Capacity Market band (`K1..K4`) and $\Delta_S$ headroom to maintain the **83% statutory discount (`K1`)**.
- `sensor.pl_hems_mpc_bess_setpoint_w` & `binary_sensor.pl_hems_zero_export_guard` — 15-min MILP battery command and `RCE <= 0` zero-export trigger.
- `sensor.pl_hems_sg_ready_heat_pump` — 2-relay **BWP `SG Ready` state (`1:0` Lock, `0:0` Normal, `0:1` Boost +2 K, `1:1` Forced Max Thermal Storage)**.
- `sensor.pl_hems_ocpp_ev_current_a` — **IEC 61851 / OCPP 1.6J** 3-phase Wallbox pilot current (`0 A` or `6..16 A`).

### B. Multi-Brand Hybrid Inverter Register Maps (`src/hems_mpc/hardware_profiles.py`)
Out-of-the-box Modbus TCP register definitions and Home Assistant YAML generators for the 4 dominant hybrid inverter platforms in Poland:
1. **Deye `SUN-8/10/12K-SG04LP3-EU`** (Solarman V5 / RS485-ETH: Reg `142` TOU Selling, Reg `154` Slot Power W, Reg `166` Target SoC %, Reg `104` Zero-Export Power, Reg `143` Max Solar Sell).
2. **Huawei `SUN2000-10KTL-M1/MAP0 + LUNA2000`** (SDongle TCP: Reg `47086` Working Mode, Reg `47100` Forcible Charge/Discharge, Reg `47075` Forced Power W, Reg `47415` Export Limit Mode).
3. **Sungrow `SH10RT + SBR096`** (WiNet-S / LAN: Reg `13049` EMS Mode, Reg `13050` Charge/Discharge Command `0xAA/0xBB/0xCC`, Reg `13051` Forced Power W, Reg `31221` Export Limit).
4. **Victron Energy `MultiPlus-II + Cerbo GX (Venus OS ESS)`** (Modbus TCP Unit 100: Reg `2700` ESS Grid Setpoint W, Reg `2702` Disable Feed-in, Reg `2901` ESS Min SoC).

### C. Virtual Power Plant (VPP) Fleet Aggregation (`250 HA Homes + 15 C&I Sites -> PSE aFRR & Intraday`)

![Home Assistant Live Entities and VPP Fleet Overview](docs/img/ha_vpp_overview.png)

A single home (`5 kW` inverter) cannot bid directly into the Polish PSE balancing market (`>= 1.0 MW` minimum bid size). In `src/hems_mpc/vpp_fleet.py`, we model a **VPP Aggregator pooling `250` Home Assistant prosumers (`1.25 MW / 2.50 MWh`) and `15` C&I B23 sites (`0.75 MW / 1.50 MWh`) = `2.00 MW / 4.00 MWh` total BESS** (plus `3.6 MW` of flexible heat pumps and EV chargers):
- Accounting for `97.2%` LTE/Wi-Fi telemetry uptime and non-coincident EV plug-in diversity, the fleet delivers **`0.96 MW` upward / `1.06 MW` downward firm `P90` reserve** (`1.35 MW` / `1.72 MW` at `P50`).
- Stacking **PSE aFRR capacity availability (`661k PLN/yr`)** and **15-min Intraday spread capture (`31k PLN/yr`)** generates **`692,204 PLN/yr` in gross VPP revenue** — paying **`+1,253 PLN/yr` in extra VPP bonus per household** (raising total household savings to **`3,530 PLN/yr`**) and **`+15,118 PLN/yr` per C&I site** after a `22%` aggregator platform margin.

> **Architecture & Commercial IP Note:** This public repository provides the open-source HIL verification testbed, HACS bridge, and baseline Monte Carlo P90 benchmark. Production ADMM distributed decomposition weights, empirical LFP cell-impedance thermal derating surfaces, and live TGE/PSE order-book bidding heuristics are maintained in the private commercial integration package.

---

## Repository Structure

```text
pl-hems-mpc/
├── data/
│   └── pl_hems_inputs_2025_2026.parquet   # 35,040 x 15-min real PSE RCE, TGE SDAC & ERA5 weather (2025-10..2026-09)
├── docs/
│   ├── index.html                         # Self-contained interactive Prosumer + C&I + Home Assistant YAML web app
│   └── img/                               # Publication-grade diagnostic plots
├── results/
│   ├── prosumer_summary.csv               # 15-scenario annual net-billing & degradation summary table
│   ├── ci_b23_summary.csv                 # C&I B23 + PSE Capacity Fee (K1..K4) summary table
│   ├── modbus_hil_trace.csv               # 672-step closed-loop Modbus TCP HIL register trace
│   ├── ha_config_example.yaml             # Ready-to-paste Home Assistant modbus: & automation: configuration
│   ├── mqtt_discovery.json                # Home Assistant MQTT Auto-Discovery payloads
│   └── summary.json                       # Complete machine-readable pipeline output
├── src/hems_mpc/
│   ├── tariffs.py                         # Polish holiday calendar, G11/G12w/Dynamic/B23 tariffs, FIFO net-billing & K-factor
│   ├── physics.py                         # Rooftop PV (Faiman + Hay sky model), 2R2C floor slab + HP COP, EV load profiles
│   ├── controllers.py                     # 15-min sparse HiGHS MILP solver + 5 prosumer controllers + C&I K1 shaper
│   ├── modbus_hil.py                      # Pure-Python Modbus TCP server/client (FC 0x03, 0x06, 0x10) + inverter watchdog
│   ├── ha_bridge.py                       # Home Assistant MQTT discovery & YAML configuration generator
│   ├── pipeline.py                        # End-to-end simulation, HIL verification, chart & HTML builder
│   └── cli.py                             # Command-line entry point
└── tests/
    └── test_hems_mpc.py                   # Unit & TCP socket integration tests
```

---

## Quickstart

```bash
git clone https://github.com/kwwaleszczynski-del/pl-hems-mpc.git
cd pl-hems-mpc
pip install -e ".[dev]"

# Run unit & Modbus TCP Hardware-in-the-Loop socket tests
pytest -v

# Re-run the full 35,040-step pipeline, HIL verification, charts, and interactive dashboard
python -m hems_mpc.cli
```

---

## License

MIT License — see [LICENSE](LICENSE).
