"""
Unit test for Optical Yellow Phase Detector with Temporal Debounce & Lockout Cooldown.
"""

import os
import sys
import numpy as np
import pytest

CORE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)

from src.perception.phase_monitor import OpticalPhaseMonitor


def test_optical_phase_monitor_debounce_and_cooldown():
    roi = {"ymin": 100, "xmin": 100, "ymax": 200, "xmax": 200}
    monitor = OpticalPhaseMonitor(
        lamp_roi=roi,
        hue_range=(18, 35),
        sat_min=120,
        val_min=150,
        active_ratio_threshold=0.25,
        debounce_count=3,
        cooldown_seconds=15.0,
    )

    # Frame with black ROI (no light)
    black_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Frame with yellow light in ROI (BGR: Yellow is (0, 255, 255))
    yellow_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    yellow_frame[100:200, 100:200] = (0, 255, 255)

    base_time = 1000.0

    # 1. Non-yellow frames: Should return False
    assert monitor.process_frame(black_frame, current_timestamp=base_time) is False
    assert monitor.consecutive_active_frames == 0

    # 2. Yellow frame 1 (debounce count = 1 / 3): Should return False
    assert monitor.process_frame(yellow_frame, current_timestamp=base_time + 0.1) is False
    assert monitor.consecutive_active_frames == 1

    # 3. Yellow frame 2 (debounce count = 2 / 3): Should return False
    assert monitor.process_frame(yellow_frame, current_timestamp=base_time + 0.2) is False
    assert monitor.consecutive_active_frames == 2

    # 4. Yellow frame 3 (debounce count = 3 / 3): Rising edge trigger should fire True!
    assert monitor.process_frame(yellow_frame, current_timestamp=base_time + 0.3) is True
    assert monitor.is_yellow_active is True
    assert monitor.total_triggers == 1

    # 5. Yellow frame 4 (continuous yellow): Should return False (already active, rising edge only)
    assert monitor.process_frame(yellow_frame, current_timestamp=base_time + 0.4) is False
    assert monitor.total_triggers == 1

    # 6. Signal turns off (transition to green/red)
    assert monitor.process_frame(black_frame, current_timestamp=base_time + 2.0) is False
    assert monitor.is_yellow_active is False
    assert monitor.consecutive_active_frames == 0

    # 7. Attempt new yellow trigger during 15s cooldown (e.g. at 1005s, only 5s passed)
    assert monitor.process_frame(yellow_frame, current_timestamp=base_time + 5.0) is False
    assert monitor.process_frame(yellow_frame, current_timestamp=base_time + 5.1) is False
    assert monitor.process_frame(yellow_frame, current_timestamp=base_time + 5.2) is False  # Debounce reached but in cooldown!
    assert monitor.total_triggers == 1

    # 8. New yellow trigger AFTER 15s cooldown expires (e.g. at 1016s)
    monitor.process_frame(black_frame, current_timestamp=base_time + 15.5)
    assert monitor.process_frame(yellow_frame, current_timestamp=base_time + 16.0) is False
    assert monitor.process_frame(yellow_frame, current_timestamp=base_time + 16.1) is False
    assert monitor.process_frame(yellow_frame, current_timestamp=base_time + 16.2) is True  # Fired!
    assert monitor.total_triggers == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
