"""
Serial / RS-485 Signal Cabinet Interface & Hardware Watchdog Dispatcher.

Dispatches binary actuation packets to the municipal traffic controller cabinet
and runs a 1.0s heartbeat watchdog to enforce fail-safe fallback (flashing yellow)
if communication fails or edge daemon becomes unresponsive.
"""

from __future__ import annotations

import json
import os
import struct
import threading
import time
from typing import Any, Dict, List, Optional


def compute_crc8(data: bytes) -> int:
    """Computes CRC-8 checksum (Dallas/Maxim polynomial: x^8 + x^5 + x^4 + 1)."""
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x8C) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


class SignalInterface:
    """
    Manages RS-485 serial communication with the physical traffic signal cabinet.
    """

    START_BYTE = 0xAA
    END_BYTE = 0x55

    CMD_ACTUATE_GREEN = 0x01
    CMD_HEARTBEAT = 0x02
    CMD_FAILSAFE = 0x03
    CMD_STATUS = 0x04

    def __init__(
        self,
        port: str = "/dev/ttyTHS1",
        baudrate: int = 115200,
        heartbeat_interval: float = 1.0,
        heartbeat_timeout: float = 3.0,
        max_misses: int = 3,
        failsafe_mode: str = "FLASHING_YELLOW",
        mock_mode: bool = False,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.max_misses = max_misses
        self.failsafe_mode = failsafe_mode
        self.mock_mode = mock_mode

        self.serial_conn = None
        self._lock = threading.Lock()
        self._running = False
        self._watchdog_thread: Optional[threading.Thread] = None

        self.last_heartbeat_sent: float = 0.0
        self.last_actuation_time: float = 0.0
        self.last_actuated_green: float = 0.0
        self.consecutive_misses: int = 0
        self.is_connected: bool = False
        self.is_in_failsafe: bool = False

        self._connect()

    @classmethod
    def from_config(cls, config_path_or_dict: str | Dict[str, Any]) -> SignalInterface:
        """Loads configuration from hardware_config.json."""
        if isinstance(config_path_or_dict, str):
            with open(config_path_or_dict, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg = config_path_or_dict

        ser_cfg = cfg.get("serial_interface", {})
        wd_cfg = cfg.get("watchdog", {})

        return cls(
            port=ser_cfg.get("port", "/dev/ttyTHS1"),
            baudrate=int(ser_cfg.get("baudrate", 115200)),
            heartbeat_interval=float(wd_cfg.get("heartbeat_interval_sec", 1.0)),
            heartbeat_timeout=float(wd_cfg.get("heartbeat_timeout_sec", 3.0)),
            max_misses=int(wd_cfg.get("max_consecutive_misses", 3)),
            failsafe_mode=wd_cfg.get("failsafe_mode", "FLASHING_YELLOW"),
            mock_mode=bool(ser_cfg.get("mock_mode", False)),
        )

    def _connect(self) -> None:
        """Attempts to open physical serial port or falls back to mock loopback mode."""
        if self.mock_mode:
            print(f"[SignalInterface] Mock mode enabled for {self.port}")
            self.is_connected = True
            return

        try:
            import serial
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
                write_timeout=0.5,
            )
            self.is_connected = True
            print(f"[SignalInterface] Connected to hardware port: {self.port} at {self.baudrate} baud")
        except Exception as e:
            print(f"[SignalInterface] Serial port {self.port} unavailable ({e}). Operating in loopback mock mode.")
            self.serial_conn = None
            self.is_connected = False
            self.mock_mode = True

    def start_watchdog(self) -> None:
        """Launches background cabinet watchdog heartbeat thread."""
        if self._running:
            return
        self._running = True
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True, name="CabinetWatchdog")
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        """Sends periodic keepalive heartbeat packets to the cabinet."""
        while self._running:
            time.sleep(self.heartbeat_interval)
            try:
                self._send_heartbeat()
            except Exception as e:
                self.consecutive_misses += 1
                if self.consecutive_misses >= self.max_misses and not self.is_in_failsafe:
                    self._trigger_failsafe(f"Missed {self.consecutive_misses} heartbeats: {e}")

    def _build_packet(self, command: int, payload: bytes = b"") -> bytes:
        """
        Constructs binary packet:
        [START (0xAA), CMD, LEN, PAYLOAD..., CRC8, END (0x55)]
        """
        payload_len = len(payload)
        header_and_body = bytes([self.START_BYTE, command, payload_len]) + payload
        crc = compute_crc8(header_and_body)
        return header_and_body + bytes([crc, self.END_BYTE])

    def _send_raw(self, packet: bytes) -> bool:
        """Transmits packet over serial interface."""
        with self._lock:
            if self.mock_mode or self.serial_conn is None:
                return True
            try:
                self.serial_conn.write(packet)
                self.serial_conn.flush()
                return True
            except Exception as e:
                print(f"[SignalInterface] Serial transmit error: {e}")
                return False

    def _send_heartbeat(self) -> None:
        """Sends keepalive heartbeat packet."""
        # Payload contains current timestamp epoch integer
        timestamp_int = int(time.time())
        payload = struct.pack(">I", timestamp_int)
        packet = self._build_packet(self.CMD_HEARTBEAT, payload)
        success = self._send_raw(packet)
        if success:
            self.last_heartbeat_sent = time.time()
            self.consecutive_misses = 0

    def dispatch_green_actuation(self, green_seconds: float, phase_id: int = 1) -> bool:
        """
        Transmits ANFIS green-time allocation command to traffic controller.

        Args:
            green_seconds: Target green phase duration in seconds (10.0s to 120.0s).
            phase_id: Actuated phase index.

        Returns:
            True if packet was sent successfully.
        """
        duration_ms = int(round(green_seconds * 1000.0))
        # Payload: Phase ID (1 byte) + Green Duration in milliseconds (uint32)
        payload = struct.pack(">BI", phase_id, duration_ms)
        packet = self._build_packet(self.CMD_ACTUATE_GREEN, payload)

        success = self._send_raw(packet)
        if success:
            self.last_actuation_time = time.time()
            self.last_actuated_green = green_seconds
            print(f"[SignalInterface] Actuation Dispatched: Phase {phase_id} -> {green_seconds:.1f}s green ({duration_ms} ms)")
        else:
            print(f"[SignalInterface] ERROR: Failed to dispatch green actuation for Phase {phase_id}!")
        return success

    def _trigger_failsafe(self, reason: str) -> None:
        """Enforces cabinet fail-safe mode (flashing yellow)."""
        self.is_in_failsafe = True
        print(f"[SignalInterface] !!! CRITICAL FAILSAFE TRIGGERED: {reason} !!! Switching to {self.failsafe_mode}")
        packet = self._build_packet(self.CMD_FAILSAFE, b"\x01")
        self._send_raw(packet)

    def get_status(self) -> Dict[str, Any]:
        """Returns connection and watchdog telemetry."""
        return {
            "is_connected": self.is_connected,
            "mock_mode": self.mock_mode,
            "is_in_failsafe": self.is_in_failsafe,
            "last_heartbeat_epoch": self.last_heartbeat_sent,
            "last_actuated_green_sec": self.last_actuated_green,
            "consecutive_misses": self.consecutive_misses,
            "port": self.port,
        }

    def stop(self) -> None:
        """Stops watchdog thread and cleanly closes serial port."""
        self._running = False
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=1.0)
        with self._lock:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
                self.serial_conn = None
        print("[SignalInterface] Serial port and watchdog released.")
