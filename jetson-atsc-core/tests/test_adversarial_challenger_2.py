"""
Adversarial Stress Test Suite - Challenger 2 (Empirical Verification)
Verifies:
1. Telemetry continuity across all 5 .jsonl traces in data/telemetry_traces/:
   - Exact frame count match with source video
   - Sequential frame_idx with ZERO gaps (0 to N-1)
   - Presence of complete actuation cycle records (t_anfis, inputs [V_w, Q, L], timestamp)
2. OpenCV video decodability across all 5 benchmark videos and all 5 comparative MP4 videos:
   - Successful decoding at start (frame 0), middle (total_frames // 2), and end (total_frames - 1)
   - Non-empty, valid frame dimensions and non-corrupt pixel content
3. Homography conditioning and reprojection accuracy across all 5 configs:
   - Condition number kappa(H) < 10^6
   - Reprojection error epsilon < 0.05m
   - Inverse conditioning kappa(H^-1) < 10^6 and inverse reprojection error < 1.0px
"""

import json
import os
import sys
import cv2
import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TELEMETRY_BENCHMARK_PAIRS = [
    {
        "name": "Night Arterial / Glare",
        "trace": "data/telemetry_traces/trace_stress_night_arterial.jsonl",
        "video": "data/sample_videos/stress_test/night_glare.mp4",
        "config": "configs/stress_night_arterial.json",
    },
    {
        "name": "Rain & Wet Road",
        "trace": "data/telemetry_traces/trace_stress_rain_wet_road.jsonl",
        "video": "data/sample_videos/stress_test/rain_wet.mp4",
        "config": "configs/stress_rain_wet_road.json",
    },
    {
        "name": "Congestion Gridlock",
        "trace": "data/telemetry_traces/trace_stress_congestion_gridlock.jsonl",
        "video": "data/sample_videos/stress_test/congestion_gridlock.mp4",
        "config": "configs/stress_congestion_gridlock.json",
    },
    {
        "name": "Bandung Pasteur ATCS",
        "trace": "data/telemetry_traces/trace_stress_indonesia_bandung_pasteur.jsonl",
        "video": "data/sample_videos/multi_angle_test/video_bandung_pasteur_h264.mp4",
        "config": "configs/stress_indonesia_bandung_pasteur.json",
    },
    {
        "name": "Bekasi ATCS",
        "trace": "data/telemetry_traces/trace_stress_indonesia_bekasi.jsonl",
        "video": "data/sample_videos/multi_angle_test/video_bekasi_atcs.mp4",
        "config": "configs/stress_indonesia_bekasi.json",
    },
]

COMPARATIVE_VIDEOS = [
    "data/comparative_media/compare_stress_night_arterial.mp4",
    "data/comparative_media/compare_stress_rain_wet_road.mp4",
    "data/comparative_media/compare_stress_congestion_gridlock.mp4",
    "data/comparative_media/compare_stress_indonesia_bandung_pasteur.mp4",
    "data/comparative_media/compare_stress_indonesia_bekasi.mp4",
]

BENCHMARK_VIDEOS = [
    "data/sample_videos/stress_test/night_glare.mp4",
    "data/sample_videos/stress_test/rain_wet.mp4",
    "data/sample_videos/stress_test/congestion_gridlock.mp4",
    "data/sample_videos/multi_angle_test/video_bandung_pasteur_h264.mp4",
    "data/sample_videos/multi_angle_test/video_bekasi_atcs.mp4",
]

CONFIG_FILES = [
    "configs/stress_night_arterial.json",
    "configs/stress_rain_wet_road.json",
    "configs/stress_congestion_gridlock.json",
    "configs/stress_indonesia_bandung_pasteur.json",
    "configs/stress_indonesia_bekasi.json",
]


class TestAdversarialChallenger2:
    """Empirical verification suite for Challenger 2."""

    # =========================================================================
    # 1. Telemetry Continuity & Completeness
    # =========================================================================

    @pytest.mark.parametrize("item", TELEMETRY_BENCHMARK_PAIRS, ids=lambda x: x["name"])
    def test_telemetry_trace_frame_count_exact_match(self, item):
        """Assert exact frame count match between telemetry trace and source video."""
        trace_path = os.path.join(PROJECT_ROOT, item["trace"])
        video_path = os.path.join(PROJECT_ROOT, item["video"])

        assert os.path.isfile(trace_path), f"Trace file missing: {trace_path}"
        assert os.path.isfile(video_path), f"Video file missing: {video_path}"

        # Read trace records
        with open(trace_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        trace_frame_count = len(records)
        assert trace_frame_count > 0, f"Trace {trace_path} is empty"

        # Determine video frame count via OpenCV
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"Cannot open video {video_path}"
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        assert trace_frame_count == video_frame_count, (
            f"Frame count mismatch for {item['name']}: "
            f"trace has {trace_frame_count} frames, video has {video_frame_count} frames"
        )

    @pytest.mark.parametrize("item", TELEMETRY_BENCHMARK_PAIRS, ids=lambda x: x["name"])
    def test_telemetry_trace_sequential_frame_idx_zero_gaps(self, item):
        """Assert sequential frame_idx with ZERO gaps from 0 to N-1."""
        trace_path = os.path.join(PROJECT_ROOT, item["trace"])
        with open(trace_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        for expected_idx, record in enumerate(records):
            assert "frame_idx" in record, f"Missing 'frame_idx' at line {expected_idx} in {trace_path}"
            actual_idx = record["frame_idx"]
            assert actual_idx == expected_idx, (
                f"Gap or out-of-order frame_idx in {trace_path} at line {expected_idx}: "
                f"expected {expected_idx}, got {actual_idx}"
            )

    @pytest.mark.parametrize("item", TELEMETRY_BENCHMARK_PAIRS, ids=lambda x: x["name"])
    def test_telemetry_trace_complete_cycle_records(self, item):
        """Assert presence of complete cycle records with t_anfis, inputs, and timestamp."""
        trace_path = os.path.join(PROJECT_ROOT, item["trace"])
        with open(trace_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        cycle_records = [r for r in records if "decision" in r]
        assert len(cycle_records) >= 1, f"No complete cycle records with 'decision' in {trace_path}"

        for idx, rec in enumerate(cycle_records):
            dec = rec["decision"]
            assert "t_anfis" in dec, f"Missing t_anfis in decision #{idx} of {trace_path}"
            assert "inputs" in dec, f"Missing inputs in decision #{idx} of {trace_path}"
            assert "timestamp" in dec, f"Missing timestamp in decision #{idx} of {trace_path}"

            t_anfis = dec["t_anfis"]
            assert isinstance(t_anfis, (int, float)), f"t_anfis not numeric in {trace_path}: {t_anfis}"
            assert 10.0 <= t_anfis <= 120.0, f"t_anfis {t_anfis} outside safe bounds [10.0, 120.0]"

            inputs = dec["inputs"]
            assert isinstance(inputs, list) and len(inputs) == 3, f"Inputs {inputs} not 3-vector in {trace_path}"
            v_w, q, l = inputs
            assert v_w >= 0.0, f"Negative flow V_w: {v_w}"
            assert q >= 0.0, f"Negative queue Q: {q}"
            assert 0.0 <= l <= 100.0, f"Occupancy L out of bounds: {l}"

    # =========================================================================
    # 2. Video Decodability at Start, Middle, and End Frames
    # =========================================================================

    @pytest.mark.parametrize("video_rel_path", BENCHMARK_VIDEOS, ids=lambda x: os.path.basename(x))
    def test_benchmark_video_decodability_start_mid_end(self, video_rel_path):
        """Check OpenCV decoding across benchmark videos at start, middle, and end frames."""
        video_path = os.path.join(PROJECT_ROOT, video_rel_path)
        assert os.path.isfile(video_path), f"Benchmark video missing: {video_path}"

        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"OpenCV failed to open benchmark video: {video_path}"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert total_frames > 0, f"Invalid total frames {total_frames} for {video_path}"

        indices_to_test = [0, total_frames // 2, total_frames - 1]
        for frame_idx in indices_to_test:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            assert ret is True, f"Failed to decode frame {frame_idx}/{total_frames} in {video_path}"
            assert frame is not None, f"Frame {frame_idx} is None in {video_path}"
            assert frame.shape[0] == 1080 and frame.shape[1] == 1920, (
                f"Unexpected frame shape {frame.shape} at frame {frame_idx} in {video_path}"
            )
            # Ensure frame has non-zero pixel data (not empty blank buffer)
            assert np.mean(frame) > 1.0, f"Frame {frame_idx} in {video_path} is completely blank/black"

        cap.release()

    @pytest.mark.parametrize("video_rel_path", COMPARATIVE_VIDEOS, ids=lambda x: os.path.basename(x))
    def test_comparative_video_decodability_start_mid_end(self, video_rel_path):
        """Check OpenCV decoding across comparative split-screen MP4 videos at start, middle, and end frames."""
        video_path = os.path.join(PROJECT_ROOT, video_rel_path)
        assert os.path.isfile(video_path), f"Comparative video missing: {video_path}"

        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"OpenCV failed to open comparative video: {video_path}"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert total_frames > 0, f"Invalid total frames {total_frames} for {video_path}"

        indices_to_test = [0, total_frames // 2, total_frames - 1]
        for frame_idx in indices_to_test:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            assert ret is True, f"Failed to decode frame {frame_idx}/{total_frames} in {video_path}"
            assert frame is not None, f"Frame {frame_idx} is None in {video_path}"
            # Comparative videos are 1920x540 split screen
            assert frame.shape[0] == 540 and frame.shape[1] == 1920, (
                f"Unexpected split frame shape {frame.shape} at frame {frame_idx} in {video_path}"
            )
            # Ensure frame has non-zero pixel data
            assert np.mean(frame) > 1.0, f"Frame {frame_idx} in {video_path} is completely blank/black"

        cap.release()

    # =========================================================================
    # 3. Homography Conditioning & Reprojection Accuracy
    # =========================================================================

    @pytest.mark.parametrize("config_rel_path", CONFIG_FILES, ids=lambda x: os.path.basename(x))
    def test_homography_conditioning_and_reprojection_error(self, config_rel_path):
        """Mathematically verify condition number kappa(H) < 10^6 and reprojection error epsilon < 0.05m."""
        config_path = os.path.join(PROJECT_ROOT, config_rel_path)
        assert os.path.isfile(config_path), f"Config file missing: {config_path}"

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        src = np.array(cfg["homography"]["src_points_pixel"], dtype=np.float64)
        dst = np.array(cfg["homography"]["dst_points_ground"], dtype=np.float64)

        assert src.shape == (4, 2), f"src shape must be (4, 2), got {src.shape}"
        assert dst.shape == (4, 2), f"dst shape must be (4, 2), got {dst.shape}"

        # Compute Homography Matrix H
        H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32)).astype(np.float64)
        H_inv = cv2.getPerspectiveTransform(dst.astype(np.float32), src.astype(np.float32)).astype(np.float64)

        # 1. Condition number kappa(H) = ||H||_2 * ||H^-1||_2
        kappa_h = np.linalg.cond(H)
        assert kappa_h < 1e6, f"Condition number kappa(H)={kappa_h:.2e} >= 10^6 in {config_rel_path}"

        kappa_h_inv = np.linalg.cond(H_inv)
        assert kappa_h_inv < 1e6, f"Inverse condition number kappa(H_inv)={kappa_h_inv:.2e} >= 10^6 in {config_rel_path}"

        # 2. Forward Reprojection Error epsilon = max || H * src - dst ||
        src_homo = np.hstack([src, np.ones((4, 1), dtype=np.float64)])  # shape (4, 3)
        dst_proj_homo = (H @ src_homo.T).T  # shape (4, 3)
        dst_proj = dst_proj_homo[:, :2] / dst_proj_homo[:, 2:3]

        reprojection_errors_m = np.linalg.norm(dst_proj - dst, axis=1)
        max_epsilon_m = float(np.max(reprojection_errors_m))
        mean_epsilon_m = float(np.mean(reprojection_errors_m))

        assert max_epsilon_m < 0.05, (
            f"Forward reprojection error epsilon={max_epsilon_m:.6f}m >= 0.05m in {config_rel_path}"
        )
        assert mean_epsilon_m < 0.05, (
            f"Mean forward reprojection error={mean_epsilon_m:.6f}m >= 0.05m in {config_rel_path}"
        )

        # 3. Inverse Reprojection Error epsilon_inv = max || H_inv * dst - src ||
        dst_homo = np.hstack([dst, np.ones((4, 1), dtype=np.float64)])
        src_proj_homo = (H_inv @ dst_homo.T).T
        src_proj = src_proj_homo[:, :2] / src_proj_homo[:, 2:3]

        inv_errors_px = np.linalg.norm(src_proj - src, axis=1)
        max_inv_epsilon_px = float(np.max(inv_errors_px))
        assert max_inv_epsilon_px < 1.0, (
            f"Inverse reprojection error epsilon_inv={max_inv_epsilon_px:.4f}px >= 1.0px in {config_rel_path}"
        )
