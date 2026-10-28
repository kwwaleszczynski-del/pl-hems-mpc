"""Multi-brand hybrid inverter Modbus register profiles (Deye, Huawei, Sungrow, Victron),
SG Ready heat pump relay logic, and OCPP 1.6J EV charger current modulation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InverterProfile:
    """Vendor-specific Modbus TCP register map for hybrid inverters popular in Poland."""
    brand_id: str
    display_name: str
    default_port: int
    unit_id: int
    protocol_note: str
    registers: dict[str, dict[str, str | int]]


INVERTER_PROFILES: dict[str, InverterProfile] = {
    "deye_sg04lp3": InverterProfile(
        brand_id="deye_sg04lp3",
        display_name="Deye SUN-8/10/12K-SG04LP3-EU (3-Phase Low-Voltage Hybrid)",
        default_port=502,
        unit_id=1,
        protocol_note="Modbus TCP via Waveshare RS485-ETH (BMS/Modbus port) or Solarman V5 stick (port 8899)",
        registers={
            "tou_selling_mode": {"addr": 142, "type": "uint16", "rw": "R/W", "desc": "Time-of-Use bitmask (Bit 0=Enable, Bits 1..6=Days)"},
            "slot1_power_w": {"addr": 154, "type": "uint16", "rw": "R/W", "desc": "Active Time-Slot Max Battery Power [W]"},
            "slot1_target_soc": {"addr": 166, "type": "uint16", "rw": "R/W", "desc": "Active Time-Slot Target SoC [%] (Grid Charge if < current)"},
            "grid_charge_enable": {"addr": 172, "type": "uint16", "rw": "R/W", "desc": "Slot Grid Charge Enable (1=Allow Grid Charge, 0=PV Only)"},
            "zero_export_power_w": {"addr": 104, "type": "uint16", "rw": "R/W", "desc": "Grid Trickle Power at Zero-Export [W] (default 20 W)"},
            "max_sell_power_w": {"addr": 143, "type": "uint16", "rw": "R/W", "desc": "Dynamic Grid Export Limit [W] (0 W when RCE <= 0)"},
            "battery_soc_pct": {"addr": 588, "type": "uint16", "rw": "R", "desc": "Battery State of Charge [%]"},
            "battery_power_w": {"addr": 590, "type": "int16", "rw": "R", "desc": "Battery Output Power [W] (+Discharge, -Charge in Deye convention)"},
            "grid_total_power_w": {"addr": 625, "type": "int16", "rw": "R", "desc": "Total Grid PCC Active Power [W] (+Import, -Export)"},
        },
    ),
    "huawei_sun2000": InverterProfile(
        brand_id="huawei_sun2000",
        display_name="Huawei SUN2000-10KTL-M1 / MAP0 + LUNA2000-10-S0",
        default_port=502,
        unit_id=1,
        protocol_note="Modbus TCP via SDongleA-05 (firmware SPC126+) or SmartLogger3000",
        registers={
            "storage_working_mode": {"addr": 47086, "type": "uint16", "rw": "R/W", "desc": "2=Max Self-Consumption, 4=TOU, 5=Fully Fed to Grid"},
            "forcible_charge_discharge": {"addr": 47100, "type": "uint16", "rw": "R/W", "desc": "0=Stop, 1=Forced Charge, 2=Forced Discharge"},
            "forced_power_w": {"addr": 47075, "type": "uint32", "rw": "R/W", "desc": "Max Charge/Discharge Power [W]"},
            "export_power_limit_mode": {"addr": 47415, "type": "uint16", "rw": "R/W", "desc": "0=Unlimited, 1=Zero-Export, 5=Power Limit (W)"},
            "export_limit_kw": {"addr": 47416, "type": "int32", "rw": "R/W", "desc": "Maximum Active Power Fed to Grid [0.001 kW]"},
            "battery_soc_permille": {"addr": 37760, "type": "uint16", "rw": "R", "desc": "LUNA2000 State of Capacity [0.1 %]"},
            "battery_power_w": {"addr": 37765, "type": "int32", "rw": "R", "desc": "LUNA2000 Charge/Discharge Power [W] (+Charge, -Discharge)"},
            "power_meter_active_w": {"addr": 37113, "type": "int32", "rw": "R", "desc": "DTSU666-H Grid PCC Power [W] (+Export, -Import)"},
        },
    ),
    "sungrow_sh10rt": InverterProfile(
        brand_id="sungrow_sh10rt",
        display_name="Sungrow SH10RT + SBR096 / SBR128",
        default_port=502,
        unit_id=1,
        protocol_note="Direct Ethernet LAN port on SH10RT or WiNet-S dongle (Modbus TCP port 502)",
        registers={
            "ems_mode_selection": {"addr": 13049, "type": "uint16", "rw": "R/W", "desc": "0=Self-Consumption (Default), 2=Forced Mode (External EMS)"},
            "charge_discharge_cmd": {"addr": 13050, "type": "uint16", "rw": "R/W", "desc": "0xAA (170)=Charge, 0xBB (187)=Discharge, 0xCC (204)=Stop"},
            "forced_power_w": {"addr": 13051, "type": "uint16", "rw": "R/W", "desc": "Forced Charge/Discharge Power [W] (0..10000)"},
            "export_limit_enable": {"addr": 31220, "type": "uint16", "rw": "R/W", "desc": "0xAA=Enable Dynamic Export Limit, 0x55=Disable"},
            "export_limit_w": {"addr": 31221, "type": "uint16", "rw": "R/W", "desc": "Export Power Limit [10 W] (0 = Zero-Export)"},
            "battery_level_soc": {"addr": 13022, "type": "uint16", "rw": "R", "desc": "Battery Level SoC [0.1 %]"},
            "battery_power_w": {"addr": 13021, "type": "uint16", "rw": "R", "desc": "Battery Power Magnitude [W]"},
            "export_power_w": {"addr": 13009, "type": "int32", "rw": "R", "desc": "Grid Export/Import Power [W] (+Export, -Import)"},
        },
    ),
    "victron_cerbo_ess": InverterProfile(
        brand_id="victron_cerbo_ess",
        display_name="Victron Energy MultiPlus-II + Cerbo GX (Venus OS ESS Mode 2/3)",
        default_port=502,
        unit_id=100,
        protocol_note="Venus OS Modbus TCP server (Unit ID 100 = VeBus/System, Unit ID 225 = Battery Monitor)",
        registers={
            "ess_ac_power_setpoint_w": {"addr": 2700, "type": "int16", "rw": "R/W", "desc": "ESS Grid Power Setpoint [W] (+Import from Grid, -Export to Grid)"},
            "ess_disable_charge": {"addr": 2701, "type": "uint16", "rw": "R/W", "desc": "0=Charge Allowed, 100=Disable Charge"},
            "ess_disable_feedback": {"addr": 2702, "type": "uint16", "rw": "R/W", "desc": "0=Feed-in Allowed, 1=Zero-Export (Disable Feed-in at RCE<=0)"},
            "ess_min_soc_pct": {"addr": 2901, "type": "uint16", "rw": "R/W", "desc": "ESS Minimum SoC Limit [0.1 %] (unless grid fails)"},
            "system_battery_soc": {"addr": 843, "type": "uint16", "rw": "R", "desc": "System Battery SoC [%]"},
            "system_battery_power_w": {"addr": 842, "type": "int16", "rw": "R", "desc": "DC Battery Power [W] (+Charging, -Discharging)"},
            "ac_grid_total_w": {"addr": 820, "type": "int16", "rw": "R", "desc": "AC Grid L1+L2+L3 Power [W]"},
        },
    ),
}


def compute_sg_ready_state(
    rce_pln_mwh: float,
    total_buy_pln_mwh: float,
    pv_surplus_kw: float,
    th_buffer_kwh: float,
) -> dict[str, str | int]:
    """Compute the German/European BWP 'SG Ready' 2-relay heat pump state (Relay 1 : Relay 2).

    States supported by Panasonic Aquarea, Vaillant aroTHERM, Nibe S-Series, Bosch Compress, Stiebel Eltron:
      - `1:0` (State 1 - EVU Lock / Peak Shaving): Heat pump compressor blocked during extreme price spikes (max 2h).
      - `0:0` (State 2 - Normal Thermostat): Standard weather-compensated heating curve.
      - `0:1` (State 3 - Boost / Overheat +2 K): Recommended turn-on during cheap ToU / solar surplus.
      - `1:1` (State 4 - Forced Max Thermal Storage): Active forced run into floor slab & DHW tank at RCE <= 0.
    """
    if rce_pln_mwh <= 0.0 and pv_surplus_kw > 0.8 and th_buffer_kwh < 8.0:
        return {"state_code": "1:1", "mode_num": 4, "label": "State 4: Forced Max Thermal Storage (RCE <= 0)"}
    if (pv_surplus_kw > 0.4 or total_buy_pln_mwh < 480.0) and th_buffer_kwh < 6.0:
        return {"state_code": "0:1", "mode_num": 3, "label": "State 3: SG Ready Boost (+2.0 K Floor Slab)"}
    if total_buy_pln_mwh > 920.0 and th_buffer_kwh > 0.3:
        return {"state_code": "1:0", "mode_num": 1, "label": "State 1: Peak Price Compressor Lock (Coast on Slab)"}
    return {"state_code": "0:0", "mode_num": 2, "label": "State 2: Normal Weather-Compensated Curve"}


def compute_ocpp_ev_current_a(ev_target_kw: float, phases: int = 3, v_ln: float = 230.0) -> int:
    """Convert MPC EV power setpoint [kW] into IEC 61851 / OCPP 1.6J pilot current [A] (0 or 6..16 A)."""
    if ev_target_kw < 1.2:
        return 0
    raw_a = (ev_target_kw * 1000.0) / (phases * v_ln)
    return int(max(6, min(16, round(raw_a))))


def render_vendor_ha_yaml(brand_id: str, host: str = "192.168.1.50", port: int | None = None) -> str:
    """Render a vendor-specific Home Assistant `modbus:` + `SG Ready` + `OCPP` configuration snippet."""
    prof = INVERTER_PROFILES.get(brand_id, INVERTER_PROFILES["sungrow_sh10rt"])
    use_port = port or prof.default_port
    regs = prof.registers
    first_rw = next((v for v in regs.values() if "W" in str(v["rw"])), list(regs.values())[0])
    soc_reg = next((v for k, v in regs.items() if "soc" in k), list(regs.values())[-1])
    grid_reg = next((v for k, v in regs.items() if "grid" in k or "meter" in k or "export_power" in k), list(regs.values())[-1])

    return f"""# ==============================================================================
# Home Assistant Hardware Integration Pack generated by pl-hems-mpc
# Target Inverter : {prof.display_name}
# Protocol Note   : {prof.protocol_note}
# ==============================================================================
modbus:
  - name: {prof.brand_id}_hub
    type: tcp
    host: {host}
    port: {use_port}
    sensors:
      - name: "{prof.brand_id.upper()} Battery SoC"
        unit_of_measurement: "%"
        slave: {prof.unit_id}
        address: {soc_reg['addr']}
        data_type: {soc_reg['type']}
        scan_interval: 15
      - name: "{prof.brand_id.upper()} Grid PCC Active Power"
        unit_of_measurement: "W"
        slave: {prof.unit_id}
        address: {grid_reg['addr']}
        data_type: {grid_reg['type']}
        device_class: power
        scan_interval: 10

automation:
  - id: pl_hems_mpc_{prof.brand_id}_dispatch
    alias: "PL-HEMS 15-min Dispatch ({prof.display_name} + SG Ready + OCPP)"
    trigger:
      - platform: time_pattern
        minutes: "/15"
    action:
      # 1. Write MPC Battery / Export Setpoint to {prof.brand_id.upper()} (Reg {first_rw['addr']}: {first_rw['desc']})
      - service: modbus.write_register
        data:
          hub: {prof.brand_id}_hub
          unit: {prof.unit_id}
          address: {first_rw['addr']}
          value: "{{{{ states('sensor.pl_hems_mpc_bess_setpoint_w') | int(0) }}}}"
      # 2. Drive Heat Pump SG Ready Relay 1 & 2 (Boost when RCE <= 0 or cheap ToU window)
      - service: switch.turn_{{{{ 'on' if is_state('binary_sensor.pl_hems_sg_ready_boost', 'on') else 'off' }}}}
        target:
          entity_id: switch.heat_pump_sg_ready_relay_2
      # 3. Modulate OCPP 1.6J Wallbox Current (0 A or 6..16 A 3-phase)
      - service: number.set_value
        target:
          entity_id: number.wallbox_ocpp_max_current
        data:
          value: "{{{{ states('sensor.pl_hems_ocpp_ev_current_a') | int(0) }}}}"
"""
