"""
Optical Yellow Phase Visual Trigger with Temporal Debounce & Lockout Cooldown.

Monitors the physical upstream traffic signal head via HSV color-space slicing
to detect the onset of the yellow clearance interval, triggering the ANFIS
control cycle precisely at peak queue accumulation.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np


class OpticalPhaseMonitor:
    """
    State machine and visual detector for traffic signal phase transitions.

    Attributes:
        lamp_roi: Dict with 'ymin', 'xmin', 'ymax', 'xmax' bounding coordinates.
        hue_range: (min_hue, max_hue) in OpenCV HSV [0, 179].
        sat_min: Minimum saturation threshold [0, 255].
        val_min: Minimum brightness/value threshold [0, 255].
        active_ratio_threshold: Minimum yellow pixel density to qualify as active (default 0.25).
        debounce_target: Consecutive frames required to confirm phase transition (default 3).
        cooldown_duration: Minimum seconds required between consecutive rising-edge triggers (default 15.0s).
    """

    def __init__(
        self,
        lamp_roi: Dict[str, int],
        hue_range: Tuple[int, int] = (18, 35),
        sat_min: int = 120,
        val_min: int = 150,
        active_ratio_threshold: float = 0.25,
        debounce_count: int = 3,
        cooldown_seconds: float = 15.0,
    ) -> None:
        self.lamp_roi = lamp_roi
        self.hue_range = hue_range
        self.sat_min = sat_min
        self.val_min = val_min
        self.active_ratio_threshold = active_ratio_threshold
        self.debounce_target = debounce_count
        self.cooldown_duration = cooldown_seconds

        # State Tracking
        self.consecutive_active_frames: int = 0
        self.is_yellow_active: bool = False
        self.last_trigger_timestamp: float = -1.0
        self.last_active_ratio: float = 0.0
        self.total_triggers: int = 0

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> OpticalPhaseMonitor:
        """Instantiates OpticalPhaseMonitor directly from intersection_roi.json."""
        lamp_cfg = config_dict.get("traffic_light_monitor", {})
        roi = lamp_cfg.get("lamp_roi_pixel", {"ymin": 120, "xmin": 1500, "ymax": 240, "xmax": 1620})
        hsv_cfg = lamp_cfg.get("hsv_thresholds", {})
        hue_range = (int(hsv_cfg.get("hue_min", 18)), int(hsv_cfg.get("hue_max", 35)))
        sat_min = int(hsv_cfg.get("sat_min", 120))
        val_min = int(hsv_cfg.get("val_min", 150))
        active_ratio = float(lamp_cfg.get("active_ratio_threshold", 0.25))
        debounce = int(lamp_cfg.get("debounce_frame_count", 3))
        cooldown = float(lamp_cfg.get("cooldown_seconds", 15.0))

        return cls(
            lamp_roi=roi,
            hue_range=hue_range,
            sat_min=sat_min,
            val_min=val_min,
            active_ratio_threshold=active_ratio,
            debounce_count=debounce,
            cooldown_seconds=cooldown,
        )

    def process_frame(self, rectified_frame: np.ndarray, current_timestamp: Optional[float] = None) -> bool:
        """
        Inspects the traffic light ROI for optical yellow phase activation.

        Args:
            rectified_frame: Lens-flattened BGR image (1080, 1920, 3).
            current_timestamp: Monotonic or system timestamp (defaults to time.time()).

        Returns:
            True strictly on rising-edge transition (debounced yellow detected AND cooldown satisfied).
        """
        if current_timestamp is None:
            current_timestamp = time.time()

        ymin = max(0, self.lamp_roi["ymin"])
        xmin = max(0, self.lamp_roi["xmin"])
        ymax = min(rectified_frame.shape[0], self.lamp_roi["ymax"])
        xmax = min(rectified_frame.shape[1], self.lamp_roi["xmax"])

        crop = rectified_frame[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            self.last_active_ratio = 0.0
            return False

        # 1. Convert crop to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # 2. Slice yellow spectrum
        lower_bound = np.array([self.hue_range[0], self.sat_min, self.val_min], dtype=np.uint8)
        upper_bound = np.array([self.hue_range[1], 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # 3. Calculate active pixel ratio
        active_pixels = cv2.countNonZero(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        active_ratio = float(active_pixels) / float(total_pixels) if total_pixels > 0 else 0.0
        self.last_active_ratio = active_ratio

        is_currently_yellow = active_ratio >= self.active_ratio_threshold

        # 4. Temporal Debouncing
        if is_currently_yellow:
            self.consecutive_active_frames += 1
        else:
            self.consecutive_active_frames = 0
            self.is_yellow_active = False

        # 5. Rising Edge Trigger Evaluation with Lockout Cooldown
        time_since_last_trigger = current_timestamp - self.last_trigger_timestamp
        in_cooldown = (self.last_trigger_timestamp > 0) and (time_since_last_trigger < self.cooldown_duration)

        rising_edge_triggered = False

        if (
            self.consecutive_active_frames >= self.debounce_target
            and not self.is_yellow_active
            and not in_cooldown
        ):
            # Fire rising-edge trigger!
            self.is_yellow_active = True
            self.last_trigger_timestamp = current_timestamp
            self.total_triggers += 1
            rising_edge_triggered = True

        return rising_edge_triggered

    def get_status(self) -> Dict[str, Any]:
        """Returns internal state telemetry for logging and debugging."""
        now = time.time()
        time_since = now - self.last_trigger_timestamp if self.last_trigger_timestamp > 0 else 9999.0
        return {
            "active_ratio": round(self.last_active_ratio, 4),
            "consecutive_frames": self.consecutive_active_frames,
            "is_yellow_active": self.is_yellow_active,
            "in_cooldown": time_since < self.cooldown_duration,
            "cooldown_remaining_sec": max(0.0, round(self.cooldown_duration - time_since, 2)),
            "total_triggers": self.total_triggers,
        }

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draws lamp ROI bounding box and live detector diagnostics onto the frame."""
        ymin = self.lamp_roi["ymin"]
        xmin = self.lamp_roi["xmin"]
        ymax = self.lamp_roi["ymax"]
        xmax = self.lamp_roi["xmax"]

        status = self.get_status()
        if self.is_yellow_active:
            color = (0, 255, 255)  # Bright Yellow BGR
            text = f"PHASE: YELLOW ACTIVE ({status['active_ratio']:.2f})"
        elif status["in_cooldown"]:
            color = (0, 165, 255)  # Orange BGR
            text = f"COOLDOWN: {status['cooldown_remaining_sec']:.1f}s"
        else:
            color = (0, 255, 0)  # Green BGR
            text = f"PHASE: MONITORING ({status['active_ratio']:.2f})"

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(
            frame,
            text,
            (xmin - 40, max(20, ymin - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        return frame
