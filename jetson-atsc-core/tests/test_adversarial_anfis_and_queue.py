"""
Adversarial Verification Suite for Sugeno ANFIS Controller & Temporal Persistence Gate.

Challenger 1 Verification:
1. Sugeno ANFIS bounds [10.0s, 120.0s] against extreme inputs (negative, zero, extreme high, NaN/inf).
2. Dynamic green scaling activation frontiers and monotonic progression.
3. Temporal persistence gate (N_min >= 5) suppressing transient glare/reflections (1, 2, 4 frames).
"""

import math
import os
import sys
import numpy as np
import pytest

CORE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)

from src.control.anfis_inference import ANFISInferenceEngine
from src.control.decision_engine import ATSCDecisionEngine
from src.common.types import TrafficSnapshot
from src.perception.detector_tracker import TrackedVehicle, DetectorTracker
from src.analytics.homography_engine import HomographyEngine
from src.analytics.traffic_metrics import TrafficMetricsExtractor


@pytest.fixture(scope="module")
def anfis_engine():
    model_path = os.path.join(CORE_ROOT, "models/anfis_sugeno.onnx")
    assert os.path.exists(model_path), f"ONNX model missing at: {model_path}"
    return ANFISInferenceEngine(
        model_path=model_path,
        prefer_cuda=False,
        min_green_sec=10.0,
        max_green_sec=120.0,
    )


@pytest.fixture(scope="module")
def homography_and_extractor():
    src_pixels = [
        [520.0, 480.0],
        [1400.0, 480.0],
        [1850.0, 1040.0],
        [70.0, 1040.0],
    ]
    dst_ground = [
        [0.0, 45.0],
        [10.5, 45.0],
        [10.5, 0.0],
        [0.0, 0.0],
    ]
    homography = HomographyEngine(src_pixels, dst_ground, ground_width_m=10.5, ground_length_m=45.0)
    extractor = TrafficMetricsExtractor(
        homography=homography,
        physical_road_area_m2=472.5,
        stopped_speed_threshold_mps=1.0,
        stop_line_tripwire_y=980,
        min_stopped_observations=5,
    )
    return homography, extractor


# ==============================================================================
# 1. Sugeno ANFIS Bounds Stress-Testing
# ==============================================================================

@pytest.mark.parametrize(
    "case_name,vw,q,l",
    [
        ("zero_traffic", 0.0, 0.0, 0.0),
        ("negative_flow", -10.0, 0.0, 0.0),
        ("negative_queue", 0.0, -25.0, 0.0),
        ("negative_occupancy", 0.0, 0.0, -50.0),
        ("all_negative", -50.0, -100.0, -80.0),
        ("extreme_high_uniform", 1e6, 1e6, 1e6),
        ("extreme_high_flow_only", 1e6, 0.0, 0.0),
        ("extreme_high_queue_only", 0.0, 1e6, 0.0),
        ("extreme_high_occupancy_only", 0.0, 0.0, 1e6),
        ("nan_flow", np.nan, 20.0, 15.0),
        ("nan_queue", 10.0, np.nan, 15.0),
        ("nan_occupancy", 10.0, 20.0, np.nan),
        ("all_nan", np.nan, np.nan, np.nan),
        ("pos_inf_flow", np.inf, 20.0, 15.0),
        ("pos_inf_queue", 10.0, np.inf, 15.0),
        ("pos_inf_occupancy", 10.0, 20.0, np.inf),
        ("neg_inf_flow", -np.inf, 20.0, 15.0),
        ("neg_inf_queue", 10.0, -np.inf, 15.0),
        ("neg_inf_occupancy", 10.0, 20.0, -np.inf),
    ],
)
def test_anfis_extreme_and_invalid_inputs(anfis_engine, case_name, vw, q, l):
    """
    Adversarial challenge: Verify that under extreme, negative, infinite,
    and NaN inputs, the Sugeno ANFIS controller strictly satisfies safe bounds [10.0s, 120.0s].
    """
    t_green = anfis_engine.evaluate_green_time(vw, q, l)

    # Must be valid finite float
    assert isinstance(t_green, float), f"Expected float, got {type(t_green)}"
    assert not math.isnan(t_green), f"Case '{case_name}' produced NaN output"
    assert not math.isinf(t_green), f"Case '{case_name}' produced Inf output"

    # Strict physical safety bounds [10.0s, 120.0s]
    assert t_green >= 10.0, f"Case '{case_name}' underflow: {t_green}s < 10.0s"
    assert t_green <= 120.0, f"Case '{case_name}' overflow: {t_green}s > 120.0s"


def test_decision_engine_bounds_wrapper(anfis_engine):
    """
    Verifies that ATSCDecisionEngine correctly flags clamp boundaries.
    """
    decision_engine = ATSCDecisionEngine(
        anfis_engine=anfis_engine,
        min_green_sec=10.0,
        max_green_sec=120.0,
    )

    # Empty intersection -> clamped to min
    dec_min = decision_engine.evaluate(
        TrafficSnapshot(v_w_pcu=0.0, queue_meters=0.0, occupancy_pct=0.0), cycle_id=1
    )
    assert dec_min.green_seconds == 10.0
    assert dec_min.is_clamped_min is True
    assert dec_min.is_clamped_max is False

    # Extreme traffic -> clamped to max
    dec_max = decision_engine.evaluate(
        TrafficSnapshot(v_w_pcu=100.0, queue_meters=45.0, occupancy_pct=100.0), cycle_id=2
    )
    assert dec_max.green_seconds == 120.0
    assert dec_max.is_clamped_max is True
    assert dec_max.is_clamped_min is False


# ==============================================================================
# 2. Dynamic Green Scaling & Monotonicity
# ==============================================================================

def test_dynamic_green_scaling_above_baseline(anfis_engine):
    """
    Adversarial challenge: Verify dynamic scaling above baseline (t_ANFIS > 10.0s).
    
    Empirical finding:
    Sugeno ANFIS incorporates a minimum clearance safety buffer of 10.0s for small isolated
    demand (e.g. 1 car: V_w=1.0, Q=0, L=1.7% -> t=10.0s).
    When demand reaches platoon level (V_w >= 8.0 PCU as observed in Bekasi ATCS Cycle 1),
    green time actively scales above baseline (t_ANFIS = 29.22s > 10.0s).
    """
    # 1. Zero demand baseline
    t_zero = anfis_engine.evaluate_green_time(0.0, 0.0, 0.0)
    assert t_zero == 10.0

    # 2. Sub-threshold demand (1 isolated car: V_w=1.0, Q=0, L=1.7%)
    # Standard traffic engineering: 10.0s is sufficient for startup lost time and clearance
    t_isolated = anfis_engine.evaluate_green_time(1.0, 0.0, 1.7)
    assert t_isolated == 10.0

    # 3. Bekasi ATCS Cycle 1 Real-World Platoon Demand (V_w=8.0 PCU, L=19.05%)
    t_bekasi = anfis_engine.evaluate_green_time(8.0, 0.0, 19.05)
    assert t_bekasi > 10.0, f"Expected dynamic scaling > 10.0s, got {t_bekasi}s"
    assert abs(t_bekasi - 29.22) < 0.1, f"Mismatch with telemetry: expected ~29.22s, got {t_bekasi}s"

    # 4. Moderate Congestion (V_w=25.0 PCU, Q=35.0m, L=20.0%)
    t_moderate = anfis_engine.evaluate_green_time(25.0, 35.0, 20.0)
    assert t_moderate > 30.0, f"Expected t_ANFIS > 30.0s for moderate congestion, got {t_moderate}s"

    # 5. Severe Gridlock (V_w=60.0 PCU, Q=45.0m, L=95.0%)
    t_gridlock = anfis_engine.evaluate_green_time(60.0, 45.0, 95.0)
    assert t_gridlock == 120.0, f"Expected maximum ceiling 120.0s, got {t_gridlock}s"

    # 6. Monotonicity validation
    assert t_zero <= t_isolated <= t_bekasi <= t_moderate <= t_gridlock


def test_dynamic_scaling_activation_thresholds(anfis_engine):
    """
    Stress-test the exact activation boundaries for V_w, Q, and L.
    Confirms that above activation thresholds, t_ANFIS strictly exceeds 10.0s.
    """
    # Flow alone activation (activation at ~8.7 PCU)
    assert anfis_engine.evaluate_green_time(8.0, 0.0, 0.0) == 10.0
    assert anfis_engine.evaluate_green_time(10.0, 0.0, 0.0) > 10.0

    # Queue alone activation (activation at ~19.1m)
    assert anfis_engine.evaluate_green_time(0.0, 15.0, 0.0) == 10.0
    assert anfis_engine.evaluate_green_time(0.0, 25.0, 0.0) > 10.0

    # Occupancy alone activation (activation at ~44.4%)
    assert anfis_engine.evaluate_green_time(0.0, 0.0, 40.0) == 10.0
    assert anfis_engine.evaluate_green_time(0.0, 0.0, 50.0) > 10.0


# ==============================================================================
# 3. Temporal Persistence Gate & Glare / Wet Reflection Suppression
# ==============================================================================

@pytest.mark.parametrize("obs_frames", [1, 2, 4])
def test_transient_glare_suppressed(homography_and_extractor, obs_frames):
    """
    Adversarial challenge: Simulate 1-frame, 2-frame, and 4-frame transient detections
    (headlight glare / wet pavement specular reflection) with zero velocity.
    Verify they do NOT qualify as stationary queue (Q = 0.0m).
    """
    _, extractor = homography_and_extractor

    # Transient glare artifact located in the middle of approach corridor (y_pixel=700)
    # Velocity is 0.0 m/s (stationary specular flash)
    transient_vehicle = TrackedVehicle(
        track_id=999,
        class_id=1,
        class_name="car",
        bbox=[900.0, 600.0, 1020.0, 700.0],
        confidence=0.88,
        timestamp=100.0 + obs_frames * 0.033,
        obs_count=obs_frames,
    )
    transient_vehicle.velocity_mps = 0.0

    # Direct property check
    assert transient_vehicle.is_stopped is False, (
        f"Transient vehicle with obs_count={obs_frames} improperly marked as is_stopped=True"
    )

    # Metric extraction check
    metrics = extractor.extract_approach_metrics([transient_vehicle])
    assert metrics["Q"] == 0.0, (
        f"False stationary queue triggered by {obs_frames}-frame glare: Q = {metrics['Q']}m (expected 0.0m)"
    )
    assert metrics["stopped_vehicles_count"] == 0, (
        f"Expected 0 stopped vehicles for {obs_frames}-frame glare, got {metrics['stopped_vehicles_count']}"
    )


def test_persistent_vehicle_qualifies_as_queue(homography_and_extractor):
    """
    Verify that genuine stopped vehicles with N >= 5 persistent observations DO qualify
    as a stationary queue (Q > 0.0m).
    """
    homography, extractor = homography_and_extractor

    persistent_vehicle = TrackedVehicle(
        track_id=101,
        class_id=1,
        class_name="car",
        bbox=[900.0, 600.0, 1020.0, 700.0],
        confidence=0.92,
        timestamp=100.0 + 5 * 0.033,
        obs_count=5,
    )
    persistent_vehicle.velocity_mps = 0.0

    # Must be marked stopped
    assert persistent_vehicle.is_stopped is True

    metrics = extractor.extract_approach_metrics([persistent_vehicle])
    _, expected_y = homography.pixel_to_ground(960.0, 700.0)

    assert metrics["stopped_vehicles_count"] == 1
    assert metrics["Q"] > 0.0
    assert abs(metrics["Q"] - expected_y) < 0.1


def test_multi_frame_tracker_glare_suppression_lifecycle(homography_and_extractor):
    """
    End-to-end multi-frame simulation using DetectorTracker.update_velocities().
    A transient headlight glare track appears for 4 frames, then disappears.
    Confirms Q remains 0.0m throughout the entire transient artifact lifecycle.
    """
    homography, extractor = homography_and_extractor

    class MockTracker:
        def __init__(self):
            self.track_trajectories = {}
            self.track_obs_counts = {}
            self.max_history_len = 10
            self.last_seen_timestamp = 0.0
        update_velocities = DetectorTracker.update_velocities

    tracker = MockTracker()
    track_id = 777

    # Frames 1 through 4: Glare artifact exists with stationary velocity
    for frame_idx in range(1, 5):
        t = 100.0 + frame_idx * 0.033
        v = TrackedVehicle(
            track_id=track_id,
            class_id=1,
            class_name="car",
            bbox=[920.0, 620.0, 1040.0, 720.0],
            confidence=0.75,
            timestamp=t,
            obs_count=tracker.track_obs_counts.get(track_id, 0),
        )
        # Update velocities in tracker
        tracker.update_velocities([v], homography.pixel_to_ground, timestamp=t)

        assert v.obs_count == frame_idx
        assert v.is_stopped is False

        metrics = extractor.extract_approach_metrics([v])
        assert metrics["Q"] == 0.0, f"Frame {frame_idx}: False queue triggered Q={metrics['Q']}"
        assert metrics["stopped_vehicles_count"] == 0

    # Frame 5: Glare vanished
    t_vanished = 100.0 + 5 * 0.033
    tracker.update_velocities([], homography.pixel_to_ground, timestamp=t_vanished)
    empty_metrics = extractor.extract_approach_metrics([])
    assert empty_metrics["Q"] == 0.0
    assert empty_metrics["stopped_vehicles_count"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
