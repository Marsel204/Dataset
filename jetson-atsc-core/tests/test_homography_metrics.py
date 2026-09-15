"""
Unit tests for Homography Inverse Perspective Mapping & PKJI 2014 Traffic Metrics.
"""

import os
import sys
import numpy as np
import pytest

CORE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)

from src.perception.detector_tracker import TrackedVehicle
from src.analytics.homography_engine import HomographyEngine
from src.analytics.traffic_metrics import TrafficMetricsExtractor


def test_homography_and_metrics_calculation():
    # 4 control points in pixel space
    src_pixels = [
        [520.0, 480.0],
        [1400.0, 480.0],
        [1850.0, 1040.0],
        [70.0, 1040.0],
    ]
    # Corresponding ground points: [X_m, Y_m]
    # Y_m is distance from stop line [0, 45.0m], X_m is lateral width [0, 10.5m]
    dst_ground = [
        [0.0, 45.0],
        [10.5, 45.0],
        [10.5, 0.0],
        [0.0, 0.0],
    ]

    homography = HomographyEngine(src_pixels, dst_ground, ground_width_m=10.5, ground_length_m=45.0)

    # 1. Verify corner projection
    for (u, v), (expected_x, expected_y) in zip(src_pixels, dst_ground):
        gx, gy = homography.pixel_to_ground(u, v)
        assert abs(gx - expected_x) < 0.05, f"X mismatch: got {gx}, expected {expected_x}"
        assert abs(gy - expected_y) < 0.05, f"Y mismatch: got {gy}, expected {expected_y}"

    # 2. Verify corridor containment
    assert homography.is_in_approach_corridor(960.0, 700.0) is True   # Center of corridor
    assert homography.is_in_approach_corridor(10.0, 10.0) is False    # Outside top-left

    # 3. Test PKJI 2014 metrics calculation
    metrics_ext = TrafficMetricsExtractor(
        homography=homography,
        physical_road_area_m2=472.5,
        stopped_speed_threshold_mps=1.0,
        stop_line_tripwire_y=980,
    )

    # Create synthetic tracked vehicles inside the corridor:
    # 3 motorcycles, 2 cars, 1 truck
    # PCU: 3*0.4 + 2*1.0 + 1*1.6 = 1.2 + 2.0 + 1.6 = 4.8 PCU
    # Footprint: 3*2.0 + 2*8.0 + 1*24.0 = 6.0 + 16.0 + 24.0 = 46.0 m^2
    # Occupancy L: 46.0 / 472.5 * 100% = 9.74%
    sim_time = 100.0
    vehicles = [
        # Stopped motorcycle at y_pixel=800 (mid-approach)
        TrackedVehicle(1, 0, "motorcycle", [900, 750, 940, 800], 0.9, sim_time),
        # Stopped motorcycle at y_pixel=820
        TrackedVehicle(2, 0, "motorcycle", [950, 770, 990, 820], 0.9, sim_time),
        # Moving motorcycle at y_pixel=900 (speed 5 m/s)
        TrackedVehicle(3, 0, "motorcycle", [850, 850, 890, 900], 0.9, sim_time),
        # Stopped car at y_pixel=600 (farther upstream, defines queue)
        TrackedVehicle(4, 1, "car", [850, 500, 1050, 600], 0.9, sim_time),
        # Stopped car at y_pixel=950 (near stop line)
        TrackedVehicle(5, 1, "car", [750, 850, 950, 950], 0.9, sim_time),
        # Stopped truck at y_pixel=700
        TrackedVehicle(6, 3, "truck", [650, 550, 850, 700], 0.9, sim_time),
    ]

    # Assign speeds: vehicle 3 is moving, others are stopped
    for v in vehicles:
        v.velocity_mps = 5.0 if v.track_id == 3 else 0.2

    results = metrics_ext.extract_approach_metrics(vehicles)
    assert abs(results["V_w"] - 4.8) < 0.05
    assert results["n_mc"] == 3
    assert results["n_lv"] == 2
    assert results["n_hv"] == 1
    assert abs(results["L"] - 9.74) < 0.1
    # Queue length should be the max Y_m among stopped vehicles (vehicle 4 at y_pixel=600)
    _, expected_q = homography.pixel_to_ground(950.0, 600.0)
    assert abs(results["Q"] - expected_q) < 0.1

    # 4. Test Stop-Line Tripwire (Camera 2 discharge)
    # Vehicle moving across tripwire at y=980 (e.g. from 960 to 1000)
    v_disch = TrackedVehicle(101, 1, "car", [900, 950, 1000, 960], 0.9, sim_time)
    metrics_ext.update_discharge_tripwire([v_disch])
    assert metrics_ext.n_served == 0  # Not yet crossed

    v_disch_crossed = TrackedVehicle(101, 1, "car", [900, 990, 1000, 1010], 0.9, sim_time + 0.1)
    metrics_ext.update_discharge_tripwire([v_disch_crossed])
    assert metrics_ext.n_served == 1  # Crossed line!

    reset_count = metrics_ext.reset_discharge_cycle()
    assert reset_count == 1
    assert metrics_ext.n_served == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
