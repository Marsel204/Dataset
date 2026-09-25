"""
End-to-End Integration Test for ATSC EdgeDaemon Pipeline.
Generates synthetic test MP4 video with optical yellow transitions,
boots EdgeDaemon in headless mode, verifies actuation, and inspects field_experiment_ledger.csv.
"""

import argparse
import csv
import os
import sys
import threading
import time
import cv2
import numpy as np
import pytest

CORE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)

from edge_daemon import EdgeDaemon


def create_synthetic_test_video(output_path: str, num_frames: int = 40) -> None:
    """Creates synthetic 1080p MP4 with green phase transitioning to optical yellow."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, 30.0, (1920, 1080))

    # Traffic light lamp ROI from intersection_roi.json:
    # ymin: 120, xmin: 1500, ymax: 240, xmax: 1620
    for f_idx in range(num_frames):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Draw corridor road background
        cv2.rectangle(frame, (400, 400), (1500, 1080), (40, 40, 40), -1)

        # Draw simulated car bbox inside corridor
        cv2.rectangle(frame, (850, 700), (1050, 950), (180, 180, 180), -1)

        # Frame 0 to 10: Green light (no yellow in lamp ROI)
        if f_idx < 10:
            cv2.circle(frame, (1560, 180), 35, (0, 255, 0), -1)  # Green circle
        # Frame 10 to 120: Optical Yellow light (HSV: Yellow BGR is (0, 255, 255))
        else:
            cv2.rectangle(frame, (1480, 100), (1640, 260), (0, 255, 255), -1)  # Solid yellow in ROI

        writer.write(frame)

    writer.release()


def test_edge_daemon_integration(tmp_path):
    test_video_path = str(tmp_path / "test_synth.mp4")
    test_ledger_path = str(tmp_path / "test_ledger.csv")
    create_synthetic_test_video(test_video_path, num_frames=120)

    args = argparse.Namespace(
        mode="SINGLE_CAM",
        source_sys=test_video_path,
        source_acq=test_video_path,
        config=os.path.join(CORE_ROOT, "configs/intersection_roi.json"),
        hardware_config=os.path.join(CORE_ROOT, "configs/hardware_config.json"),
        ledger=test_ledger_path,
        imgsz=320,  # fast resolution for testing
        headless=True,
        loop=True,
        mock_hardware=True,
    )

    daemon = EdgeDaemon(args)
    daemon.auto_thermal_fallback = False  # Disable auto-recovery to avoid /dev/video1 probing

    # Run daemon in background thread
    daemon_thread = threading.Thread(target=daemon.start, daemon=True)
    daemon_thread.start()

    # Poll dynamically for yellow actuation cycle
    t_start = time.time()
    while time.time() - t_start < 45.0:
        if daemon.cycle_count >= 1:
            break
        time.sleep(0.1)

    daemon.is_running = False
    daemon_thread.join(timeout=3.0)

    # Verify ledger was created and populated
    assert os.path.exists(test_ledger_path), "Ledger CSV was not generated."
    with open(test_ledger_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\n[E2E Integration] Generated {len(rows)} cycle records in test ledger.")
    assert len(rows) >= 1, "Expected at least one yellow phase actuation record."

    record = rows[0]
    assert int(record["cycle_id"]) == 1
    assert float(record["t_anfis_sec"]) >= 10.0
    assert float(record["t_anfis_sec"]) <= 120.0
    assert record["watchdog_status"] == "OK"
    print(f"[E2E Integration SINGLE_CAM] Actuated Green Time: {record['t_anfis_sec']}s (V_w={record['v_w_pcu']}, Q={record['q_meters']}m)")


def test_edge_daemon_dual_cam_integration(tmp_path):
    test_video_path = str(tmp_path / "test_synth_dual.mp4")
    test_ledger_path = str(tmp_path / "test_ledger_dual.csv")
    create_synthetic_test_video(test_video_path, num_frames=120)

    args = argparse.Namespace(
        mode="DUAL_CAM",
        source_sys=test_video_path,
        source_acq=test_video_path,
        config=os.path.join(CORE_ROOT, "configs/intersection_roi.json"),
        hardware_config=os.path.join(CORE_ROOT, "configs/hardware_config.json"),
        ledger=test_ledger_path,
        imgsz=320,
        headless=True,
        loop=True,
        mock_hardware=True,
    )

    daemon = EdgeDaemon(args)
    daemon.auto_thermal_fallback = False

    daemon_thread = threading.Thread(target=daemon.start, daemon=True)
    daemon_thread.start()

    t_start = time.time()
    while time.time() - t_start < 20.0:
        if daemon.cycle_count >= 1:
            break
        time.sleep(0.1)

    daemon.is_running = False
    daemon_thread.join(timeout=3.0)

    assert os.path.exists(test_ledger_path)
    with open(test_ledger_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) >= 1
    assert rows[0]["mode"] == "DUAL_CAM"
    print(f"[E2E Integration DUAL_CAM] Actuated Green Time: {rows[0]['t_anfis_sec']}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
