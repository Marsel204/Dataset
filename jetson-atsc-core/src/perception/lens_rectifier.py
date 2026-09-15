"""
Zero-Latency Lens Rectifier for Brica B-PRO5 Alpha 170-degree Fisheye Action Camera.

Applies precomputed Kannala-Brandt unwarping maps via cv2.remap to eliminate
barrel distortion without runtime projection math overhead.
"""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple
import cv2
import numpy as np


class LensRectifier:
    """
    High-performance optical distortion correction engine.

    Loads Kannala-Brandt K & D distortion matrices from an NPZ archive,
    precomputes pixel-coordinate remap lookup tables once at initialization,
    and performs fast, zero-latency bilinear unwarping during real-time inference.
    """

    def __init__(
        self,
        calib_path: str,
        balance: float = 0.0,
        fov_scale: float = 1.0,
    ) -> None:
        """
        Initialize the lens rectifier.

        Args:
            calib_path: Path to brica_fisheye_calib.npz containing 'K', 'D', 'dim'.
            balance: Lens balance factor [0.0, 1.0]. 0.0 retains all pixels, 1.0 fills the frame.
            fov_scale: Scaling parameter for the new camera matrix.
        """
        self.calib_path = calib_path
        self.balance = balance
        self.fov_scale = fov_scale

        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"Calibration file not found at: {calib_path}")

        calib = np.load(calib_path)
        self.K: np.ndarray = calib["K"]
        self.D: np.ndarray = calib["D"]
        self.dim: Tuple[int, int] = tuple(calib["dim"])  # (width, height)

        # Precompute rectification maps
        self.map1, self.map2 = self._build_remap_tables()
        self.last_rectify_ms: float = 0.0

    def _build_remap_tables(self) -> Tuple[np.ndarray, np.ndarray]:
        """Precomputes cv2.fisheye undistortion lookup tables."""
        w, h = self.dim
        # Calculate optimal new camera matrix to prevent severe border cropping
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K,
            self.D,
            (w, h),
            np.eye(3),
            balance=self.balance,
            fov_scale=self.fov_scale,
        )

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K,
            self.D,
            np.eye(3),
            new_K,
            (w, h),
            cv2.CV_16SC2,  # Fixed-point 16-bit representation for maximum cv2.remap speed
        )
        return map1, map2

    def rectify(self, frame: np.ndarray) -> np.ndarray:
        """
        Undistorts an incoming frame using precomputed cv2.remap.

        Args:
            frame: Input BGR frame of resolution (1080, 1920, 3)

        Returns:
            Distortion-flattened BGR frame of identical resolution
        """
        t0 = time.perf_counter()
        undistorted = cv2.remap(
            frame,
            self.map1,
            self.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        self.last_rectify_ms = (time.perf_counter() - t0) * 1000.0
        return undistorted


if __name__ == "__main__":
    # Self-test and latency benchmark
    test_calib = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../configs/brica_fisheye_calib.npz")
    )
    rectifier = LensRectifier(test_calib)

    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.rectangle(dummy_frame, (100, 100), (1820, 980), (255, 255, 255), 4)

    # Warmup
    for _ in range(5):
        _ = rectifier.rectify(dummy_frame)

    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = rectifier.rectify(dummy_frame)
        times.append((time.perf_counter() - t0) * 1000.0)

    print(f"[LensRectifier] 1080p cv2.remap average latency: {np.mean(times):.2f} ms")
