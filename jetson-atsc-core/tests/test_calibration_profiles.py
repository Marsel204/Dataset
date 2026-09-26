"""
Unit & Integration Test Suite for ATSC Camera Calibration Profiles & IPM Homography.

Validates the 5 stress and Indonesian calibration profiles against the 7 physical
validity criteria defined in ORIGINAL_REQUEST.md, PROJECT.md, and survey reports:
1. Non-Degenerate Quad Convexity
2. Non-Negative Metric Dimensions & Schema Ordering
3. Standard Arterial Geometry & Physical Area Consistency
4. Homography Invertibility & Conditioning (kappa(H) < 1e6)
5. Bidirectional Reprojection Accuracy (epsilon < 0.05m, epsilon_inv < 1.0px)
6. Model & Lens Calibration Asset Resolution
7. PKJI 2014 Passenger Car Unit (PCE) Standard Weights & Actuation Configuration
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

from src.analytics.homography_engine import HomographyEngine
from src.analytics.traffic_metrics import TrafficMetricsExtractor

MANDATORY_CONFIG_FILES = [
    "configs/stress_night_arterial.json",
    "configs/stress_rain_wet_road.json",
    "configs/stress_congestion_gridlock.json",
    "configs/stress_indonesia_bandung_pasteur.json",
    "configs/stress_indonesia_bekasi.json",
]

ALIAS_CONFIG_FILES = [
    "configs/stress_night_glare.json",
    "configs/stress_rain_wet.json",
]

ALL_CONFIG_FILES = MANDATORY_CONFIG_FILES + ALIAS_CONFIG_FILES


@pytest.fixture(params=ALL_CONFIG_FILES)
def config_data(request):
    """Loads configuration JSON file."""
    rel_path = request.param
    full_path = os.path.join(PROJECT_ROOT, rel_path)
    assert os.path.exists(full_path), f"Configuration file not found: {rel_path}"
    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return rel_path, data


class TestCalibrationProfiles:
    """Rigorous physical validity test suite for camera calibration configurations."""

    def test_criterion_1_quad_convexity_and_bounds(self, config_data):
        """
        Criterion 1: Non-Degenerate Quad Convexity.
        The 4 source control points must reside within image boundaries (1920x1080),
        have non-zero positive area, and form a strictly convex non-self-intersecting quadrilateral.
        """
        rel_path, data = config_data
        src = np.array(data["homography"]["src_points_pixel"], dtype=np.float32)

        # 4 control points
        assert src.shape == (4, 2), f"{rel_path}: src_points_pixel must have shape (4, 2)"

        # Within 1080p frame bounds [0, 1920] x [0, 1080]
        assert np.all(src[:, 0] >= 0.0) and np.all(src[:, 0] <= 1920.0), (
            f"{rel_path}: u coordinates out of bounds [0, 1920]: {src[:, 0]}"
        )
        assert np.all(src[:, 1] >= 0.0) and np.all(src[:, 1] <= 1080.0), (
            f"{rel_path}: v coordinates out of bounds [0, 1080]: {src[:, 1]}"
        )

        # Positive contour area
        area = cv2.contourArea(src)
        assert area > 1000.0, f"{rel_path}: Quad area too small or degenerate: {area} px^2"

        # Convexity: Cross product of consecutive edge vectors must maintain uniform sign
        edges = np.roll(src, -1, axis=0) - src
        cross_z = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
        is_strictly_convex = np.all(cross_z > 0) or np.all(cross_z < 0)
        assert is_strictly_convex, f"{rel_path}: Polygon is concave or self-intersecting: cross_z={cross_z}"

    def test_criterion_2_metric_bounds_and_schema_ordering(self, config_data):
        """
        Criterion 2: Non-Negative Metric Dimensions & Standard Coordinate Ordering.
        Ground points must strictly satisfy:
          Index 0 (TL): [0.0, L]
          Index 1 (TR): [W, L]
          Index 2 (BR): [W, 0.0]
          Index 3 (BL): [0.0, 0.0]
        with W in [7.0, 12.0]m, L in (0.0, 75.0]m, and all ground coordinates >= 0.0.
        """
        rel_path, data = config_data
        dst = np.array(data["homography"]["dst_points_ground"], dtype=np.float32)
        dims = data["homography"]["ground_dimensions_m"]
        w = float(dims["width"])
        length = float(dims["length"])

        assert dst.shape == (4, 2), f"{rel_path}: dst_points_ground must have shape (4, 2)"
        assert np.all(dst >= 0.0), f"{rel_path}: Negative ground distances detected: {dst}"

        # Standard canonical ordering verification
        expected_dst = np.array([
            [0.0, length],
            [w, length],
            [w, 0.0],
            [0.0, 0.0],
        ], dtype=np.float32)

        np.testing.assert_allclose(
            dst, expected_dst, atol=1e-4,
            err_msg=f"{rel_path}: dst_points_ground does not match canonical [[0, L], [W, L], [W, 0], [0, 0]]"
        )

    def test_criterion_3_arterial_geometry_and_area_consistency(self, config_data):
        """
        Criterion 3: Standard Arterial Geometry & Physical Area Consistency.
        - Corridor width W between 7.0m and 12.0m (2 to 3.5 lanes at 3.5m/lane).
        - Upstream reach L <= 75.0m (Requirement R2 maximum reach).
        - physical_road_area_m2 == round(width * length, 2).
        """
        rel_path, data = config_data
        dims = data["homography"]["ground_dimensions_m"]
        w = float(dims["width"])
        length = float(dims["length"])
        reported_area = float(data["lane_metrics"]["physical_road_area_m2"])

        assert 7.0 <= w <= 12.0, f"{rel_path}: Corridor width {w}m outside standard arterial bounds [7.0, 12.0]m"
        assert 10.0 <= length <= 75.0, f"{rel_path}: Corridor reach {length}m outside bounds [10.0, 75.0]m"

        expected_area = round(w * length, 2)
        assert abs(reported_area - expected_area) < 0.01, (
            f"{rel_path}: Area inconsistency: reported {reported_area}m^2 vs computed {expected_area}m^2"
        )

    def test_criterion_4_homography_invertibility_and_conditioning(self, config_data):
        """
        Criterion 4: Homography Matrix Invertibility & Numerical Conditioning.
        - det(H) != 0 and det(H_inv) != 0.
        - Condition number kappa(H) < 1e6 (ensuring numerical stability against pixel noise).
        """
        rel_path, data = config_data
        src = np.array(data["homography"]["src_points_pixel"], dtype=np.float32)
        dst = np.array(data["homography"]["dst_points_ground"], dtype=np.float32)

        H = cv2.getPerspectiveTransform(src, dst)
        H_inv = cv2.getPerspectiveTransform(dst, src)

        det_h = float(np.linalg.det(H))
        det_h_inv = float(np.linalg.det(H_inv))
        assert abs(det_h) > 1e-12, f"{rel_path}: Homography matrix is singular (det={det_h})"
        assert abs(det_h_inv) > 1e-12, f"{rel_path}: Inverse homography is singular (det={det_h_inv})"

        cond_h = float(np.linalg.cond(H))
        cond_h_inv = float(np.linalg.cond(H_inv))
        assert cond_h < 1e6, f"{rel_path}: Homography condition number kappa(H)={cond_h:.2e} exceeds 1e6 threshold"
        assert cond_h_inv < 1e6, f"{rel_path}: Inverse homography condition number kappa(H_inv)={cond_h_inv:.2e} exceeds 1e6 threshold"

    def test_criterion_5_bidirectional_reprojection_accuracy(self, config_data):
        """
        Criterion 5: Bidirectional Reprojection Accuracy.
        - Forward projection error: max || H * src - dst || < 0.05m.
        - Inverse projection error: max || H_inv * dst - src || < 1.0 pixel.
        """
        rel_path, data = config_data
        src = np.array(data["homography"]["src_points_pixel"], dtype=np.float32)
        dst = np.array(data["homography"]["dst_points_ground"], dtype=np.float32)

        H = cv2.getPerspectiveTransform(src, dst)
        H_inv = cv2.getPerspectiveTransform(dst, src)

        # Forward reprojection (pixels -> meters)
        src_reshaped = src.reshape(-1, 1, 2)
        fwd_projected = cv2.perspectiveTransform(src_reshaped, H).reshape(-1, 2)
        fwd_errors_m = np.linalg.norm(fwd_projected - dst, axis=1)
        max_fwd_err = float(np.max(fwd_errors_m))
        assert max_fwd_err < 0.05, f"{rel_path}: Forward reprojection error {max_fwd_err:.4e}m >= 0.05m"

        # Inverse reprojection (meters -> pixels)
        dst_reshaped = dst.reshape(-1, 1, 2)
        inv_projected = cv2.perspectiveTransform(dst_reshaped, H_inv).reshape(-1, 2)
        inv_errors_px = np.linalg.norm(inv_projected - src, axis=1)
        max_inv_err = float(np.max(inv_errors_px))
        assert max_inv_err < 1.0, f"{rel_path}: Inverse reprojection error {max_inv_err:.4e}px >= 1.0px"

    def test_criterion_6_lens_calibration_and_model_path_resolution(self, config_data):
        """
        Criterion 6: Lens Calibration & Model Path Pairing.
        - system.calibration_path points to a valid rectilinear calibration file.
        - system.model_path points to an existing YOLO model or checkpoint.
        """
        rel_path, data = config_data
        sys_cfg = data.get("system", {})

        model_path = sys_cfg.get("model_path")
        calib_path = sys_cfg.get("calibration_path")

        assert model_path, f"{rel_path}: Missing system.model_path"
        assert calib_path, f"{rel_path}: Missing system.calibration_path"

        # Verify paths exist on disk (relative to project root or absolute)
        abs_model = model_path if os.path.isabs(model_path) else os.path.join(PROJECT_ROOT, model_path)
        abs_calib = calib_path if os.path.isabs(calib_path) else os.path.join(PROJECT_ROOT, calib_path)

        assert os.path.exists(abs_model), f"{rel_path}: Model file not found at {abs_model}"
        assert os.path.exists(abs_calib), f"{rel_path}: Calibration file not found at {abs_calib}"

    def test_criterion_7_pcu_weights_and_actuation_parameters(self, config_data):
        """
        Criterion 7: PKJI 2014 Passenger Car Unit (PCE) Standards & Actuation Configuration.
        - pcu_weights must conform to PKJI 2014: motorcycle=0.4, car=1.0, truck=1.6, bus=1.6.
        - active_ratio_threshold must be 0.99 for non-optical fallback trigger feeds.
        - corridor_polygon must be present and match src_points_pixel.
        - green_time_bounds_sec must be [10.0, 120.0].
        """
        rel_path, data = config_data
        lane_cfg = data.get("lane_metrics", {})
        tlm_cfg = data.get("traffic_light_monitor", {})

        # PKJI 2014 weights
        pcu = lane_cfg.get("pcu_weights", {})
        assert pcu.get("motorcycle") == 0.4, f"{rel_path}: Expected motorcycle PCU 0.4, got {pcu.get('motorcycle')}"
        assert pcu.get("car") == 1.0, f"{rel_path}: Expected car PCU 1.0, got {pcu.get('car')}"
        assert pcu.get("truck") == 1.6, f"{rel_path}: Expected truck PCU 1.6, got {pcu.get('truck')}"
        assert pcu.get("bus") == 1.6, f"{rel_path}: Expected bus PCU 1.6, got {pcu.get('bus')}"

        # Vehicle footprints
        footprints = lane_cfg.get("vehicle_footprints_m2", {})
        assert footprints.get("motorcycle") == 2.0
        assert footprints.get("car") == 8.0
        assert footprints.get("truck") == 24.0
        assert footprints.get("bus") == 24.0

        # Green time bounds
        assert lane_cfg.get("green_time_bounds_sec") == [10.0, 120.0]

        # Active ratio threshold for fallback periodic triggering compatibility
        assert tlm_cfg.get("active_ratio_threshold") == 0.99, (
            f"{rel_path}: active_ratio_threshold should be 0.99 for fallback compatibility, got {tlm_cfg.get('active_ratio_threshold')}"
        )

        # Corridor polygon must match src_points_pixel
        corridor = lane_cfg.get("corridor_polygon")
        src = data["homography"]["src_points_pixel"]
        assert corridor is not None, f"{rel_path}: lane_metrics.corridor_polygon is missing"
        np.testing.assert_allclose(corridor, src, atol=1e-4, err_msg=f"{rel_path}: corridor_polygon does not match src_points_pixel")

    def test_engine_instantiation_and_end_to_end_projection(self, config_data):
        """
        End-to-end verification: Instantiates HomographyEngine and TrafficMetricsExtractor
        directly from the configuration file and verifies point projection roundtrip.
        """
        rel_path, data = config_data
        full_path = os.path.join(PROJECT_ROOT, rel_path)

        engine = HomographyEngine.from_config(full_path)
        extractor = TrafficMetricsExtractor.from_config(data, engine)

        # Test midpoint projection
        u_mid = float(np.mean(engine.src_points[:, 0]))
        v_mid = float(np.mean(engine.src_points[:, 1]))
        assert engine.is_in_approach_corridor(u_mid, v_mid), f"{rel_path}: Midpoint not inside approach corridor"

        gx, gy = engine.pixel_to_ground(u_mid, v_mid)
        assert 0.0 <= gx <= engine.ground_width, f"{rel_path}: Projected X {gx}m outside bounds [0, {engine.ground_width}]"
        assert 0.0 <= gy <= engine.ground_length, f"{rel_path}: Projected Y {gy}m outside bounds [0, {engine.ground_length}]"

        # Roundtrip check
        u_rt, v_rt = engine.ground_to_pixel(gx, gy)
        assert abs(u_rt - u_mid) <= 2.0, f"{rel_path}: Pixel roundtrip error in u: {abs(u_rt - u_mid)}"
        assert abs(v_rt - v_mid) <= 2.0, f"{rel_path}: Pixel roundtrip error in v: {abs(v_rt - v_mid)}"
