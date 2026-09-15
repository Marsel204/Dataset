"""
Unit tests for SignalInterface, SystemTelemetry, and AsyncRecorder.
"""

import os
import shutil
import sys
import time
import numpy as np
import pytest

CORE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)

from src.control.signal_interface import SignalInterface, compute_crc8
from src.monitoring.system_telemetry import SystemTelemetry
from src.storage.async_recorder import AsyncRecorder


def test_signal_interface_packet_and_crc():
    # Verify CRC8 computation
    test_bytes = b"\xAA\x01\x05\x00\x00\x75\x30"
    crc = compute_crc8(test_bytes)
    assert 0 <= crc <= 255

    # Test SignalInterface loopback mock mode
    sig = SignalInterface(port="/dev/mock_tty", mock_mode=True)
    sig.start_watchdog()

    # Actuate 30.0s green
    success = sig.dispatch_green_actuation(green_seconds=30.0, phase_id=1)
    assert success is True
    assert sig.last_actuated_green == 30.0

    status = sig.get_status()
    assert status["mock_mode"] is True
    assert status["is_connected"] is True
    sig.stop()


def test_system_telemetry_dynamic_fallback():
    telem = SystemTelemetry(
        thermal_threshold_celsius=75.0,
        min_fps_threshold=15.0,
        thermal_recovery_celsius=68.0,
        recovery_fps_threshold=20.0,
    )

    # 1. Nominal case: Temp=50C, FPS=30 -> Maintain DUAL_CAM
    new_mode, changed, _ = telem.evaluate_fallback("DUAL_CAM", temp_c=50.0, fps=30.0)
    assert new_mode == "DUAL_CAM"
    assert changed is False

    # 2. Overheating case: Temp=76C -> Trigger fallback to SINGLE_CAM
    new_mode, changed, reason = telem.evaluate_fallback("DUAL_CAM", temp_c=76.0, fps=25.0)
    assert new_mode == "SINGLE_CAM"
    assert changed is True
    assert "Thermal degradation" in reason

    # 3. FPS collapse case: Temp=60C, FPS=12 -> Trigger fallback to SINGLE_CAM
    new_mode, changed, reason = telem.evaluate_fallback("DUAL_CAM", temp_c=60.0, fps=12.0)
    assert new_mode == "SINGLE_CAM"
    assert changed is True
    assert "Throughput degradation" in reason

    # 4. Thermal recovery case: Temp=65C, FPS=24 -> Recover back to DUAL_CAM
    new_mode, changed, reason = telem.evaluate_fallback("SINGLE_CAM", temp_c=65.0, fps=24.0)
    assert new_mode == "DUAL_CAM"
    assert changed is True
    assert "Thermal recovery" in reason


def test_async_recorder_non_blocking_write(tmp_path):
    test_dir = str(tmp_path / "test_recordings")
    recorder = AsyncRecorder(
        output_dir=test_dir,
        filename_prefix="test_cam",
        width=640,
        height=480,
        fps=30.0,
        max_buffer_size=10,
    )
    recorder.start()

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(5):
        ok = recorder.enqueue_frame(dummy_frame, f"TEST_FRAME_{i}")
        assert ok is True

    time.sleep(0.3)
    recorder.stop()

    assert recorder.recorded_frames > 0
    assert os.path.exists(recorder.current_filepath)
    assert os.path.getsize(recorder.current_filepath) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
