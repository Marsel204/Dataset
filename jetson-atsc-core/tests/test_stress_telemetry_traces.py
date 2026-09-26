"""
Validation and verification suite for stress-test telemetry traces.

Asserts:
1. All 5 benchmark .jsonl trace files exist and exceed 10KB.
2. Sequential continuity with zero missing frames (frame_idx from 0 to N-1).
3. Complete actuation cycle records with "decision" containing t_anfis, inputs [V_w, Q, L], and timestamp.
4. Sugeno ANFIS green allocations strictly adhere to safe bounds [10.0s, 120.0s].
5. Arriving flow and standing queues dynamically scale green times above baseline (t_ANFIS > 10.0s) under demand.
6. Profiling throughput confirms >= 30 FPS and end-to-end latency <= 33 ms.
7. False stationary queue suppression via temporal persistence gate (N_min >= 5).
"""

import json
import os
import sys
import pytest
import numpy as np

CORE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)

TRACE_DIR = os.path.join(CORE_ROOT, "data/telemetry_traces")

BENCHMARK_TRACES = [
    "trace_stress_night_arterial.jsonl",
    "trace_stress_rain_wet_road.jsonl",
    "trace_stress_congestion_gridlock.jsonl",
    "trace_stress_indonesia_bandung_pasteur.jsonl",
    "trace_stress_indonesia_bekasi.jsonl",
]


@pytest.mark.parametrize("trace_filename", BENCHMARK_TRACES)
def test_stress_trace_file_exists_and_size(trace_filename):
    """Assert all 5 .jsonl trace files exist and have byte size > 10KB."""
    trace_path = os.path.join(TRACE_DIR, trace_filename)
    assert os.path.isfile(trace_path), f"Telemetry trace file not found: {trace_path}"
    byte_size = os.path.getsize(trace_path)
    assert byte_size > 10 * 1024, f"Trace {trace_filename} size ({byte_size} bytes) is <= 10KB"


@pytest.mark.parametrize("trace_filename", BENCHMARK_TRACES)
def test_sequential_frame_continuity_zero_missing_frames(trace_filename):
    """Assert sequential continuity with ZERO missing frames (frame_idx from 0 to N-1)."""
    trace_path = os.path.join(TRACE_DIR, trace_filename)
    with open(trace_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    assert len(records) >= 700, f"Expected >= 700 frames in {trace_filename}, found {len(records)}"

    last_ts = -1.0
    for expected_idx, record in enumerate(records):
        assert "frame_idx" in record, f"Missing 'frame_idx' in record {expected_idx} of {trace_filename}"
        assert record["frame_idx"] == expected_idx, (
            f"Frame continuity violation in {trace_filename} at line {expected_idx}: "
            f"expected frame_idx={expected_idx}, got {record['frame_idx']}"
        )
        assert record["frame"] == expected_idx + 1, (
            f"Frame 1-based indexing violation in {trace_filename} at line {expected_idx}: "
            f"expected frame={expected_idx + 1}, got {record['frame']}"
        )
        assert record["timestamp"] >= last_ts, (
            f"Timestamp non-monotonic in {trace_filename} at frame_idx {expected_idx}: "
            f"{record['timestamp']} < {last_ts}"
        )
        last_ts = record["timestamp"]


@pytest.mark.parametrize("trace_filename", BENCHMARK_TRACES)
def test_complete_cycle_records_and_schema(trace_filename):
    """Assert every trace contains complete cycle records with 'decision' containing t_anfis, inputs, and timestamp."""
    trace_path = os.path.join(TRACE_DIR, trace_filename)
    with open(trace_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    cycle_records = [r for r in records if "decision" in r]
    assert len(cycle_records) >= 1, f"No actuation cycle records found in {trace_filename}"

    for r in cycle_records:
        dec = r["decision"]
        assert "t_anfis" in dec, f"Missing 't_anfis' in decision of {trace_filename}"
        assert "inputs" in dec, f"Missing 'inputs' in decision of {trace_filename}"
        assert "timestamp" in dec, f"Missing 'timestamp' in decision of {trace_filename}"
        assert "cycle_id" in dec, f"Missing 'cycle_id' in decision of {trace_filename}"
        assert "green_seconds" in dec, f"Missing 'green_seconds' in decision of {trace_filename}"

        # Validate inputs vector [V_w, Q, L]
        inputs = dec["inputs"]
        assert isinstance(inputs, list) and len(inputs) == 3, (
            f"Inputs vector in {trace_filename} must be 3-element list [V_w, Q, L], got {inputs}"
        )
        v_w, q, l = inputs
        assert v_w >= 0.0, f"Negative V_w: {v_w}"
        assert q >= 0.0, f"Negative Q: {q}"
        assert 0.0 <= l <= 100.0, f"Occupancy L out of bounds [0, 100]: {l}"


def test_sugeno_anfis_safe_bounds_across_all_traces():
    """Assert Sugeno ANFIS green allocations strictly adhere to safe bounds [10.0s, 120.0s]."""
    for trace_filename in BENCHMARK_TRACES:
        trace_path = os.path.join(TRACE_DIR, trace_filename)
        with open(trace_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                record = json.loads(line)
                if "decision" in record:
                    dec = record["decision"]
                    t_anfis = dec["t_anfis"]
                    assert 10.0 <= t_anfis <= 120.0, (
                        f"ANFIS allocation {t_anfis}s violates safe bounds [10.0s, 120.0s] "
                        f"in {trace_filename} at line {line_no}"
                    )
                    assert dec["green_seconds"] == t_anfis


def test_dynamic_demand_scaling_above_baseline():
    """Assert arriving flow and standing queues dynamically scale green times above baseline (> 10.0s)."""
    # 1. Empirically verify Bekasi ATCS cycle decision under high vehicle demand
    bekasi_trace = os.path.join(TRACE_DIR, "trace_stress_indonesia_bekasi.jsonl")
    with open(bekasi_trace, "r", encoding="utf-8") as f:
        bekasi_records = [json.loads(line) for line in f]

    bekasi_cycles = [r["decision"] for r in bekasi_records if "decision" in r]
    assert len(bekasi_cycles) >= 1
    # Cycle 1 in Bekasi has V_w=8.0 PCU, L=19.05% -> scaled green time
    scaled_cycle = bekasi_cycles[0]
    assert scaled_cycle["t_anfis"] > 10.0, (
        f"Expected t_anfis > 10.0s for high demand in Bekasi cycle 1, got {scaled_cycle['t_anfis']}"
    )
    assert scaled_cycle["t_anfis"] >= 25.0, (
        f"Expected t_anfis >= 25.0s for V_w=8.0 PCU, got {scaled_cycle['t_anfis']}"
    )

    # 2. Directly verify ANFIS inference engine dynamic scaling response
    from src.control.anfis_inference import ANFISInferenceEngine
    anfis = ANFISInferenceEngine(os.path.join(CORE_ROOT, "models/anfis_sugeno.onnx"))

    baseline_green = anfis.evaluate_green_time(0.0, 0.0, 0.0)
    assert baseline_green == 10.0, f"Baseline green under zero demand must be 10.0s, got {baseline_green}"

    # Increasing demand scenarios
    light_demand_green = anfis.evaluate_green_time(10.0, 20.0, 15.0)
    heavy_demand_green = anfis.evaluate_green_time(35.0, 40.0, 50.0)
    gridlock_green = anfis.evaluate_green_time(70.0, 45.0, 90.0)

    assert light_demand_green > baseline_green, f"Light demand ({light_demand_green}s) <= baseline ({baseline_green}s)"
    assert heavy_demand_green > light_demand_green, f"Heavy demand ({heavy_demand_green}s) <= light ({light_demand_green}s)"
    assert gridlock_green >= heavy_demand_green, f"Gridlock ({gridlock_green}s) < heavy ({heavy_demand_green}s)"
    assert gridlock_green <= 120.0


@pytest.mark.parametrize("trace_filename", BENCHMARK_TRACES)
def test_profiling_throughput_and_latency(trace_filename):
    """Assert profiling throughput confirms >= 30 FPS and end-to-end latency <= 33 ms."""
    trace_path = os.path.join(TRACE_DIR, trace_filename)
    with open(trace_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    # Exclude initial warmup frame
    sample_records = records[1:]
    latencies = [r["processing_time_ms"] for r in sample_records]
    fps_vals = [r["fps"] for r in sample_records]

    mean_latency = float(np.mean(latencies))
    median_latency = float(np.median(latencies))
    mean_fps = float(np.mean(fps_vals))

    # Real-time envelope: >= 30 FPS, <= 33 ms
    assert mean_latency <= 33.0, (
        f"Mean latency in {trace_filename} ({mean_latency:.2f}ms) exceeds 33ms limit"
    )
    assert median_latency <= 33.0, (
        f"Median latency in {trace_filename} ({median_latency:.2f}ms) exceeds 33ms limit"
    )
    assert mean_fps >= 30.0, (
        f"Mean throughput in {trace_filename} ({mean_fps:.1f} FPS) below 30 FPS target"
    )


def test_temporal_persistence_gate_false_queue_suppression():
    """Assert temporal persistence gate (N_min >= 5) prevents false stationary queues."""
    from src.perception.detector_tracker import TrackedVehicle
    from src.analytics.traffic_metrics import TrafficMetricsExtractor
    from src.analytics.homography_engine import HomographyEngine

    config_path = os.path.join(CORE_ROOT, "configs/stress_rain_wet_road.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    homography = HomographyEngine.from_config(config)
    metrics_ext = TrafficMetricsExtractor.from_config(config, homography)

    # 1. Transient reflection: vehicle observed for only 1 frame at zero velocity
    v_transient = TrackedVehicle(
        track_id=999,
        class_id=1,
        class_name="car",
        bbox=[500, 700, 600, 800],
        confidence=0.45,
        timestamp=1.0,
        obs_count=1,  # Transient (1 frame)
    )
    v_transient.velocity_mps = 0.0
    v_transient.ground_pos = (5.0, 35.0)  # Located 35m upstream

    # Transient detection must not qualify as stopped
    assert v_transient.is_stopped is False

    # Extract metrics with transient vehicle in corridor
    metrics = metrics_ext.extract_approach_metrics([v_transient])
    assert metrics["Q"] == 0.0, f"Transient detection at 35m incorrectly caused queue Q={metrics['Q']}"
    assert metrics["stopped_vehicles_count"] == 0

    # 2. Confirmed persistent vehicle: vehicle observed for >= 5 frames at zero velocity
    v_persistent = TrackedVehicle(
        track_id=999,
        class_id=1,
        class_name="car",
        bbox=[500, 700, 600, 800],
        confidence=0.85,
        timestamp=1.0,
        obs_count=5,  # Persistent (5 frames)
    )
    v_persistent.velocity_mps = 0.0
    v_persistent.ground_pos = (5.0, 35.0)

    assert v_persistent.is_stopped is True
    metrics_persistent = metrics_ext.extract_approach_metrics([v_persistent])
    assert metrics_persistent["Q"] == 35.0, f"Persistent vehicle failed to register queue Q={metrics_persistent['Q']}"
    assert metrics_persistent["stopped_vehicles_count"] == 1
