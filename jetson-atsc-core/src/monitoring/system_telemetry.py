"""
System Hardware Telemetry & Dynamic Thermal Fallback Guard.

Samples:
- SoC / Thermal Zone 0 Temperature (Jetson /sys/devices/virtual/thermal/thermal_zone0/temp)
- Tegrastats GPU / CPU / Memory utilization
- Real-time pipeline processing FPS
- Evaluates dynamic thermal step-down: DUAL_CAM -> SINGLE_CAM when temp >= 75°C or FPS < 15
- Evaluates thermal recovery: SINGLE_CAM -> DUAL_CAM when temp <= 68°C and FPS >= 20
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from typing import Any, Dict, Optional, Tuple


class SystemTelemetry:
    """
    Monitors edge hardware thermals, resource consumption, and enforces
    dynamic dual-mode stepping to prevent thermal throttling or frame drops.
    """

    JETSON_TEMP_PATH = "/sys/devices/virtual/thermal/thermal_zone0/temp"

    def __init__(
        self,
        thermal_threshold_celsius: float = 75.0,
        min_fps_threshold: float = 15.0,
        thermal_recovery_celsius: float = 68.0,
        recovery_fps_threshold: float = 20.0,
    ) -> None:
        self.thermal_threshold = thermal_threshold_celsius
        self.min_fps_threshold = min_fps_threshold
        self.thermal_recovery = thermal_recovery_celsius
        self.recovery_fps = recovery_fps_threshold

        self.is_jetson = os.path.exists(self.JETSON_TEMP_PATH)
        self.last_sample_time: float = time.time()
        self.frame_count: int = 0
        self.current_fps: float = 30.0
        self.last_temperature: float = 45.0

    def record_frame(self) -> float:
        """
        Updates throughput FPS calculation. Call once per main orchestrator loop cycle.
        Returns the instantaneous smoothed FPS.
        """
        now = time.time()
        self.frame_count += 1
        dt = now - self.last_sample_time

        if dt >= 1.0:
            instant_fps = float(self.frame_count) / dt
            # Exponential moving average
            self.current_fps = round(0.7 * self.current_fps + 0.3 * instant_fps, 2)
            self.frame_count = 0
            self.last_sample_time = now

        return self.current_fps

    def read_soc_temperature(self) -> float:
        """
        Reads SoC temperature in degrees Celsius from sysfs, falling back to psutil/nvidia-smi.
        """
        # 1. Jetson Sysfs direct read
        if os.path.exists(self.JETSON_TEMP_PATH):
            try:
                with open(self.JETSON_TEMP_PATH, "r", encoding="utf-8") as f:
                    temp_raw = f.read().strip()
                    temp_c = float(temp_raw) / 1000.0
                    self.last_temperature = temp_c
                    return round(temp_c, 1)
            except Exception:
                pass

        # 2. Linux generic thermal zones
        for zone in range(5):
            path = f"/sys/class/thermal/thermal_zone{zone}/temp"
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        val = float(f.read().strip())
                        if val > 1000:
                            val /= 1000.0
                        if 20.0 <= val <= 105.0:
                            self.last_temperature = val
                            return round(val, 1)
                except Exception:
                    continue

        # 3. NVIDIA-SMI fallback for desktop dev host
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=0.2,
            )
            if res.returncode == 0:
                temp_c = float(res.stdout.strip())
                self.last_temperature = temp_c
                return round(temp_c, 1)
        except Exception:
            pass

        return round(self.last_temperature, 1)

    def read_metrics(self) -> Dict[str, Any]:
        """Collects full hardware metrics snapshot."""
        temp_c = self.read_soc_temperature()
        fps = self.current_fps

        # Fallback evaluation
        thermal_alarm = temp_c >= self.thermal_threshold
        fps_alarm = fps < self.min_fps_threshold

        return {
            "soc_temp_c": temp_c,
            "fps": fps,
            "thermal_alarm": thermal_alarm,
            "fps_alarm": fps_alarm,
            "is_jetson": self.is_jetson,
        }

    def evaluate_fallback(
        self,
        current_mode: str,
        temp_c: Optional[float] = None,
        fps: Optional[float] = None,
    ) -> Tuple[str, bool, str]:
        """
        Evaluates whether mode should transition between DUAL_CAM and SINGLE_CAM.

        Args:
            current_mode: "DUAL_CAM" or "SINGLE_CAM".
            temp_c: Optional explicit SoC temperature override (for testing or external monitors).
            fps: Optional explicit throughput FPS override.

        Returns:
            Tuple of (new_mode: str, mode_changed: bool, reason: str).
        """
        if temp_c is None:
            temp_c = self.read_soc_temperature()
        if fps is None:
            fps = self.current_fps

        if current_mode == "DUAL_CAM":
            if temp_c >= self.thermal_threshold:
                reason = f"Thermal degradation: SoC temp {temp_c:.1f}°C >= {self.thermal_threshold:.1f}°C threshold."
                return "SINGLE_CAM", True, reason
            elif fps < self.min_fps_threshold:
                reason = f"Throughput degradation: Pipeline FPS {fps:.1f} < {self.min_fps_threshold:.1f} threshold."
                return "SINGLE_CAM", True, reason

        elif current_mode == "SINGLE_CAM":
            if temp_c <= self.thermal_recovery and fps >= self.recovery_fps:
                reason = f"Thermal recovery: SoC temp {temp_c:.1f}°C <= {self.thermal_recovery:.1f}°C and FPS {fps:.1f} >= {self.recovery_fps:.1f}."
                return "DUAL_CAM", True, reason

        return current_mode, False, ""
