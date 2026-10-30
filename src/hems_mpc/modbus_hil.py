"""Software-in-the-loop test rig: a Modbus TCP server and client around an emulated hybrid inverter (no hardware).

Implements the standard binary Modbus TCP framing (7-byte MBAP header + PDU) for:
  - Function Code 0x03: Read Holding Registers
  - Function Code 0x06: Write Single Holding Register
  - Function Code 0x10: Write Multiple Holding Registers

Register map (project-specific; not a vendor or SunSpec map, so it must not be written to a real inverter):
  - 40001 (uint16): EMS Control Mode (0 = Local Self-Consumption, 2 = Forced Charge/Discharge)
  - 40002 (int16) : Forced Battery Power Command [W] (+charge, -discharge, signed two's complement)
  - 40003 (uint16): Export Power Limit [W] (Zero-Export when 0)
  - 40004 (uint16): Watchdog Heartbeat Counter (incremented by EMS every control cycle)
  - 40005 (uint16): Watchdog Timeout [s] (default 900 s; reverts mode to 0 if heartbeat stops)
  - 40010 (uint16): Battery SoC [0.1 %] (0..1000)
  - 40011 (uint16): PV Available AC Power [W]
  - 40012 (uint16): Site Base Load Power [W]
  - 40013 (int16) : Actual Battery Power [W] (+charging, -discharging)
  - 40014 (int16) : Grid Point-of-Common-Coupling Power [W] (+import, -export)
  - 40015 (uint16): Curtailed PV Power [W]
  - 40016 (uint16): Watchdog Tripped Flag (0 = OK, 1 = Fallback Active)
"""

from __future__ import annotations

import socket
import struct
import threading
from dataclasses import dataclass, field


def to_uint16(val: int) -> int:
    return int(val) & 0xFFFF


def to_int16(val: int) -> int:
    v = int(val) & 0xFFFF
    return v - 0x10000 if v >= 0x8000 else v


@dataclass
class VirtualHybridInverter:
    """Deterministic physical state machine backing the Modbus TCP holding registers."""
    bess_capacity_wh: float = 10_000.0
    max_charge_w: int = 5_000
    max_discharge_w: int = 5_000
    soc_min_permille: int = 100
    soc_max_permille: int = 950
    eta_ch: float = 0.955
    eta_dis: float = 0.955
    registers: dict[int, int] = field(default_factory=dict)
    _last_heartbeat: int = 0
    _elapsed_since_hb_s: int = 0

    def __post_init__(self) -> None:
        self.registers = {
            40001: 0,      # EMS mode: 0 = self-consumption
            40002: 0,      # Forced battery power command [W]
            40003: 10_000, # Export power limit [W]
            40004: 0,      # Watchdog heartbeat counter
            40005: 900,    # Watchdog timeout [s] (15 min)
            40010: 500,    # Battery SoC = 50.0%
            40011: 0,      # PV power [W]
            40012: 500,    # Load power [W]
            40013: 0,      # Actual battery power [W]
            40014: 500,    # Grid power [W] (+import, -export)
            40015: 0,      # Curtailed PV [W]
            40016: 0,      # Watchdog tripped flag
        }
        self._last_heartbeat = self.registers[40004]

    def step_physics(self, pv_avail_w: int, load_w: int, dt_s: int = 900) -> dict[str, int]:
        """Advance the inverter & battery by `dt_s` seconds under current Modbus register commands."""
        self.registers[40011] = max(0, int(pv_avail_w))
        self.registers[40012] = max(0, int(load_w))

        hb_now = self.registers[40004]
        if hb_now != self._last_heartbeat:
            self._last_heartbeat = hb_now
            self._elapsed_since_hb_s = 0
            self.registers[40016] = 0
        else:
            self._elapsed_since_hb_s += dt_s

        # Emulated watchdog: if the EMS heartbeat stalls beyond the timeout, revert to local self-consumption
        if self._elapsed_since_hb_s > self.registers[40005] and self.registers[40001] != 0:
            self.registers[40001] = 0
            self.registers[40002] = 0
            self.registers[40003] = 10_000
            self.registers[40016] = 1

        mode = self.registers[40001]
        cmd_p_w = to_int16(self.registers[40002])
        exp_lim_w = self.registers[40003]
        soc_pm = self.registers[40010]
        dt_h = dt_s / 3600.0

        surplus_w = pv_avail_w - load_w
        if mode == 0:
            want_batt_w = surplus_w
        else:
            want_batt_w = cmd_p_w

        # Enforce SoC and power ratings
        if want_batt_w >= 0:
            max_e_wh = max(0.0, (self.soc_max_permille - soc_pm) / 1000.0 * self.bess_capacity_wh)
            max_w_by_soc = int(max_e_wh / (dt_h * self.eta_ch))
            batt_w = min(self.max_charge_w, want_batt_w, max_w_by_soc)
            d_soc_pm = int(round((batt_w * dt_h * self.eta_ch / self.bess_capacity_wh) * 1000.0))
        else:
            max_e_wh = max(0.0, (soc_pm - self.soc_min_permille) / 1000.0 * self.bess_capacity_wh)
            max_w_by_soc = int(max_e_wh * self.eta_dis / dt_h)
            batt_w = -min(self.max_discharge_w, -want_batt_w, max_w_by_soc)
            d_soc_pm = int(round((batt_w * dt_h / self.eta_dis / self.bess_capacity_wh) * 1000.0))

        soc_pm = max(self.soc_min_permille, min(self.soc_max_permille, soc_pm + d_soc_pm))

        # Apply export limit (Zero-Export / dynamic curtailment)
        raw_export_w = pv_avail_w - load_w - batt_w
        if raw_export_w > exp_lim_w:
            curt_w = raw_export_w - exp_lim_w
        else:
            curt_w = 0

        grid_w = load_w + batt_w - (pv_avail_w - curt_w)

        self.registers[40010] = to_uint16(soc_pm)
        self.registers[40013] = to_uint16(batt_w)
        self.registers[40014] = to_uint16(grid_w)
        self.registers[40015] = to_uint16(curt_w)

        return {
            "soc_permille": soc_pm,
            "batt_w": batt_w,
            "grid_w": grid_w,
            "curt_w": curt_w,
            "watchdog_tripped": self.registers[40016],
        }


def handle_modbus_pdu(inv: VirtualHybridInverter, unit_id: int, pdu: bytes) -> bytes:
    """Process a single Modbus PDU (0x03, 0x06, 0x10) against the virtual inverter registers."""
    fc = pdu[0]
    if fc == 0x03:
        start_addr, count = struct.unpack(">HH", pdu[1:5])
        vals = [to_uint16(inv.registers.get(start_addr + i, 0)) for i in range(count)]
        return struct.pack(">BB", 0x03, count * 2) + struct.pack(f">{count}H", *vals)
    elif fc == 0x06:
        reg_addr, val = struct.unpack(">HH", pdu[1:5])
        inv.registers[reg_addr] = to_uint16(val)
        return pdu[:5]
    elif fc == 0x10:
        start_addr, count, byte_cnt = struct.unpack(">HHB", pdu[1:6])
        vals = struct.unpack(f">{count}H", pdu[6 : 6 + byte_cnt])
        for i, v in enumerate(vals):
            inv.registers[start_addr + i] = to_uint16(v)
        return struct.pack(">BHH", 0x10, start_addr, count)
    else:
        return struct.pack(">BB", fc | 0x80, 0x01)


class ModbusTCPServer:
    """Lightweight threaded Modbus TCP server on localhost for software-in-the-loop testing."""

    def __init__(self, inverter: VirtualHybridInverter, host: str = "127.0.0.1", port: int = 0) -> None:
        self.inverter = inverter
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(5)
        self.host, self.port = self._sock.getsockname()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                break
            threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()

    def _handle_client(self, conn: socket.socket) -> None:
        with conn:
            while not self._stop.is_set():
                header = conn.recv(7)
                if len(header) < 7:
                    break
                tx_id, proto_id, length, unit_id = struct.unpack(">HHHB", header)
                pdu = conn.recv(length - 1)
                if not pdu:
                    break
                resp_pdu = handle_modbus_pdu(self.inverter, unit_id, pdu)
                resp_mbap = struct.pack(">HHHB", tx_id, proto_id, len(resp_pdu) + 1, unit_id)
                conn.sendall(resp_mbap + resp_pdu)


class ModbusTCPClient:
    """Minimal synchronous Modbus TCP client for reading telemetry and writing MPC setpoints."""

    def __init__(self, host: str, port: int, unit_id: int = 1, timeout: float = 2.0) -> None:
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.timeout = timeout
        self._tx_id = 0
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _exchange(self, pdu: bytes) -> bytes:
        if self._sock is None:
            self.connect()
        assert self._sock is not None
        self._tx_id = (self._tx_id + 1) & 0xFFFF
        mbap = struct.pack(">HHHB", self._tx_id, 0, len(pdu) + 1, self.unit_id)
        self._sock.sendall(mbap + pdu)
        hdr = self._sock.recv(7)
        _, _, length, _ = struct.unpack(">HHHB", hdr)
        return self._sock.recv(length - 1)

    def read_holding_registers(self, start_addr: int, count: int) -> list[int]:
        pdu = struct.pack(">BHH", 0x03, start_addr, count)
        resp = self._exchange(pdu)
        return list(struct.unpack(f">{count}H", resp[2 : 2 + count * 2]))

    def write_multiple_registers(self, start_addr: int, values: list[int]) -> None:
        count = len(values)
        u_vals = [to_uint16(v) for v in values]
        pdu = struct.pack(">BHHB", 0x10, start_addr, count, count * 2) + struct.pack(f">{count}H", *u_vals)
        self._exchange(pdu)
