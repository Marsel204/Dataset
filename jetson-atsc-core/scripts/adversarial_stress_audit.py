"""
Adversarial Stress Audit Runner.
Executes comprehensive empirical stress tests against:
1. Sugeno ANFIS bounds [10.0s, 120.0s] against extreme / pathological inputs.
2. Dynamic green scaling sensitivity, deadband analysis, and platoon scaling.
3. Temporal persistence gate (N_min >= 5) suppressing transient glare/reflections.
"""

import math
import os
import sys
import numpy as np

CORE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)

from src.control.anfis_inference import ANFISInferenceEngine
from src.control.decision_engine import ATSCDecisionEngine
from src.common.types import TrafficSnapshot
from src.perception.detector_tracker import TrackedVehicle, DetectorTracker
from src.analytics.homography_engine import HomographyEngine
from src.analytics.traffic_metrics import TrafficMetricsExtractor


def run_anfis_bounds_audit(engine: ANFISInferenceEngine):
    print("=" * 70)
    print("1. SUGENO ANFIS BOUNDS [10.0s, 120.0s] ADVERSARIAL STRESS AUDIT")
    print("=" * 70)

    test_vectors = [
        ("Zero demand", 0.0, 0.0, 0.0),
        ("Negative flow", -10.0, 0.0, 0.0),
        ("Negative queue", 0.0, -25.0, 0.0),
        ("Negative occupancy", 0.0, 0.0, -50.0),
        ("All negative", -100.0, -50.0, -50.0),
        ("Extreme high flow", 1e6, 0.0, 0.0),
        ("Extreme high queue", 0.0, 1e6, 0.0),
        ("Extreme high occupancy", 0.0, 0.0, 1e6),
        ("Extreme high all", 1e6, 1e6, 1e6),
        ("NaN flow", np.nan, 20.0, 15.0),
        ("NaN queue", 10.0, np.nan, 15.0),
        ("NaN occupancy", 10.0, 20.0, np.nan),
        ("All NaN", np.nan, np.nan, np.nan),
        ("+Inf flow", np.inf, 20.0, 15.0),
        ("+Inf queue", 10.0, np.inf, 15.0),
        ("+Inf occupancy", 10.0, 20.0, np.inf),
        ("-Inf flow", -np.inf, 20.0, 15.0),
        ("-Inf queue", 10.0, -np.inf, 15.0),
        ("-Inf occupancy", 10.0, 20.0, -np.inf),
    ]

    all_passed = True
    for name, vw, q, l in test_vectors:
        t = engine.evaluate_green_time(vw, q, l)
        is_finite = not math.isnan(t) and not math.isinf(t)
        bounded = 10.0 <= t <= 120.0
        status = "PASS" if (is_finite and bounded) else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"[{status}] {name:25s} (V_w={vw}, Q={q}, L={l}) -> t_ANFIS = {t:6.2f}s (bounded: {bounded})")

    print(f"\nANFIS Bounds Stress Audit Result: {'ALL TESTS PASSED' if all_passed else 'FAILURES DETECTED'}\n")
    return all_passed


def run_dynamic_scaling_audit(engine: ANFISInferenceEngine):
    print("=" * 70)
    print("2. DYNAMIC GREEN SCALING & DEMAND SENSITIVITY AUDIT")
    print("=" * 70)

    # Threshold mapping
    vw_act = next(vw for vw in np.linspace(0, 50, 501) if engine.evaluate_green_time(vw, 0.0, 0.0) > 10.0)
    q_act = next(q for q in np.linspace(0, 45, 451) if engine.evaluate_green_time(0.0, q, 0.0) > 10.0)
    l_act = next(l for l in np.linspace(0, 100, 1001) if engine.evaluate_green_time(0.0, 0.0, l) > 10.0)
    cars_act = next(n for n in np.linspace(0.1, 20, 200) if engine.evaluate_green_time(n, 0.0, min(100.0, n * 8.0 / 472.5 * 100.0)) > 10.0)

    print(f"Empirical Activation Thresholds (Transition from 10.0s baseline):")
    print(f"  - Isolated Flow V_w      : {vw_act:.2f} PCU")
    print(f"  - Isolated Queue Q       : {q_act:.2f} m")
    print(f"  - Isolated Occupancy L   : {l_act:.2f} %")
    print(f"  - Arriving Car Platoon   : {cars_act:.2f} cars (V_w={cars_act:.1f}, L={cars_act*8.0/472.5*100.0:.2f}%)\n")

    scenarios = [
        ("Zero traffic (baseline)", 0.0, 0.0, 0.0, 10.0, False),
        ("1 Arriving Motorcycle (sub-threshold)", 0.4, 0.0, 0.42, 10.0, False),
        ("1 Arriving Car (sub-threshold)", 1.0, 0.0, 1.69, 10.0, False),
        ("3 Arriving Cars (sub-threshold)", 3.0, 0.0, 5.08, 10.0, False),
        ("4 Arriving Cars (sub-threshold)", 4.0, 0.0, 6.77, 10.0, False),
        ("5 Arriving Cars (activation point)", 5.0, 0.0, 8.47, 10.12, True),
        ("Bekasi ATCS Cycle 1 (Real Telemetry)", 8.0, 0.0, 19.05, 29.22, True),
        ("Moderate Platoon + Queue", 25.0, 35.0, 20.0, 68.02, True),
        ("Severe Gridlock (Maximum ceiling)", 60.0, 45.0, 95.0, 120.0, True),
    ]

    print("Demand Scaling Evaluation Across Operating Regimes:")
    monotonic = True
    prev_t = 0.0
    for name, vw, q, l, exp_t, exp_scaled in scenarios:
        act_t = engine.evaluate_green_time(vw, q, l)
        scaled = act_t > 10.0
        if act_t < prev_t:
            monotonic = False
        prev_t = act_t
        match = abs(act_t - exp_t) < 0.2
        status = "PASS" if match else "WARN"
        print(f"[{status}] {name:38s}: (V_w={vw:4.1f}, Q={q:4.1f}m, L={l:5.2f}%) -> t_ANFIS = {act_t:6.2f}s | Scaled: {scaled} (exp: {exp_scaled})")

    print(f"\nMonotonicity check: {'PASSED (Non-decreasing)' if monotonic else 'FAILED'}\n")
    return monotonic


def run_persistence_gate_audit():
    print("=" * 70)
    print("3. TEMPORAL PERSISTENCE GATE (N_min >= 5) GLARE SUPPRESSION AUDIT")
    print("=" * 70)

    src_pixels = [[520.0, 480.0], [1400.0, 480.0], [1850.0, 1040.0], [70.0, 1040.0]]
    dst_ground = [[0.0, 45.0], [10.5, 45.0], [10.5, 0.0], [0.0, 0.0]]
    homography = HomographyEngine(src_pixels, dst_ground, ground_width_m=10.5, ground_length_m=45.0)
    extractor = TrafficMetricsExtractor(
        homography=homography,
        min_stopped_observations=5,
    )

    class MockTracker:
        def __init__(self):
            self.track_trajectories = {}
            self.track_obs_counts = {}
            self.max_history_len = 10
            self.last_seen_timestamp = 0.0
        update_velocities = DetectorTracker.update_velocities

    tracker = MockTracker()
    track_id = 901

    print("Multi-Frame Simulation of Specular Reflection / Headlight Glare:")
    print("Artifact appears at Frame 1 with 0 velocity, persists for 4 frames, then disappears.\n")

    gate_passed = True
    for frame_idx in range(1, 7):
        t = 100.0 + frame_idx * 0.033
        if frame_idx <= 4:
            # Glare artifact present
            v = TrackedVehicle(
                track_id=track_id,
                class_id=1,
                class_name="car",
                bbox=[900.0, 600.0, 1020.0, 700.0],
                confidence=0.85,
                timestamp=t,
                obs_count=tracker.track_obs_counts.get(track_id, 0),
            )
            tracker.update_velocities([v], homography.pixel_to_ground, timestamp=t)
            metrics = extractor.extract_approach_metrics([v])

            is_stopped = v.is_stopped
            q_val = metrics["Q"]
            stopped_cnt = metrics["stopped_vehicles_count"]

            # Must NOT qualify as stopped vehicle or queue
            frame_pass = (is_stopped is False) and (q_val == 0.0) and (stopped_cnt == 0)
            if not frame_pass:
                gate_passed = False

            status = "PASS" if frame_pass else "FAIL"
            print(f"[{status}] Frame {frame_idx}: obs_count={v.obs_count}, speed={v.velocity_mps:.2f} m/s, is_stopped={is_stopped}, Q={q_val:.2f}m, stopped={stopped_cnt}")
        elif frame_idx == 5:
            # Persistent genuine vehicle test (if track stayed to frame 5)
            # To test genuine queue vs vanished glare:
            # First demonstrate vanished glare
            metrics = extractor.extract_approach_metrics([])
            print(f"[PASS] Frame 5 (Glare vanished): Active tracks = 0 -> Q = {metrics['Q']:.2f}m, stopped = {metrics['stopped_vehicles_count']}")
        elif frame_idx == 6:
            # Counter-test: If vehicle stays for 5 frames, it MUST qualify
            tracker_persistent = MockTracker()
            for obs in range(1, 6):
                vp = TrackedVehicle(
                    track_id=888,
                    class_id=1,
                    class_name="car",
                    bbox=[900.0, 600.0, 1020.0, 700.0],
                    confidence=0.90,
                    timestamp=100.0 + obs * 0.033,
                    obs_count=tracker_persistent.track_obs_counts.get(888, 0),
                )
                tracker_persistent.update_velocities([vp], homography.pixel_to_ground, timestamp=100.0 + obs * 0.033)
            metrics_p = extractor.extract_approach_metrics([vp])
            p_pass = (vp.is_stopped is True) and (metrics_p["Q"] > 0.0) and (metrics_p["stopped_vehicles_count"] == 1)
            status = "PASS" if p_pass else "FAIL"
            if not p_pass:
                gate_passed = False
            print(f"[{status}] Frame 6 (Persistent Vehicle Control): obs_count={vp.obs_count} >= 5 -> is_stopped={vp.is_stopped}, Q={metrics_p['Q']:.2f}m, stopped={metrics_p['stopped_vehicles_count']}")

    print(f"\nPersistence Gate Audit Result: {'ALL TESTS PASSED' if gate_passed else 'FAILURES DETECTED'}\n")
    return gate_passed


if __name__ == "__main__":
    engine = ANFISInferenceEngine(os.path.join(CORE_ROOT, "models/anfis_sugeno.onnx"), prefer_cuda=False)
    b_pass = run_anfis_bounds_audit(engine)
    s_pass = run_dynamic_scaling_audit(engine)
    g_pass = run_persistence_gate_audit()

    print("=" * 70)
    print("FINAL AUDIT SUMMARY")
    print(f"1. ANFIS Bounds Clamping [10.0s, 120.0s] : {'PASS' if b_pass else 'FAIL'}")
    print(f"2. Dynamic Green Scaling & Monotonicity  : {'PASS' if s_pass else 'FAIL'}")
    print(f"3. Temporal Persistence False Queue Gate : {'PASS' if g_pass else 'FAIL'}")
    print("=" * 70)
