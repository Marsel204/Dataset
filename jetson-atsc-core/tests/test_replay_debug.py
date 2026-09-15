"""
Unit tests for deterministic ReplayDebugger CLI tool.
"""

import json
import os
import sys
import cv2
import numpy as np

CORE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)

from scripts.replay_debug import ReplayDebugger


def create_synthetic_mp4(path: str, num_frames: int = 25) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (1920, 1080))
    for f in range(num_frames):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Add yellow light in traffic light ROI for frame >= 5
        if f >= 5:
            cv2.rectangle(frame, (1480, 100), (1640, 260), (0, 255, 255), -1)
        writer.write(frame)
    writer.release()


def test_replay_debugger_headless_and_trace(tmp_path):
    video_path = str(tmp_path / "test_replay.mp4")
    trace_path = str(tmp_path / "trace.jsonl")
    create_synthetic_mp4(video_path, num_frames=15)

    debugger = ReplayDebugger(
        video_path=video_path,
        config_path=os.path.join(CORE_ROOT, "configs/intersection_roi.json"),
        calib_path=os.path.join(CORE_ROOT, "configs/brica_fisheye_calib.npz"),
        model_path=os.path.join(CORE_ROOT, "models/Final.pt"),
        anfis_path=os.path.join(CORE_ROOT, "models/anfis_sugeno.onnx"),
        headless=True,
        step_mode=False,
        max_frames=12,
        trace_path=trace_path,
        imgsz=320,
    )

    debugger.run()

    assert debugger.frame_idx == 12
    assert os.path.exists(trace_path)

    with open(trace_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 12
    first_record = json.loads(lines[0])
    assert first_record["frame"] == 1
    assert "traffic" in first_record
    assert "v_w_pcu" in first_record["traffic"]
