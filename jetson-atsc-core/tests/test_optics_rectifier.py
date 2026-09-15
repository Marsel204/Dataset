"""
Unit test for Fisheye Optics and Zero-Latency Lens Rectifier.
"""

import os
import sys
import numpy as np
import pytest

CORE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)

from src.perception.lens_rectifier import LensRectifier


def test_lens_rectifier_initialization_and_remap():
    calib_path = os.path.join(CORE_ROOT, "configs/brica_fisheye_calib.npz")
    assert os.path.exists(calib_path), f"Calibration file missing at {calib_path}"

    rectifier = LensRectifier(calib_path)
    assert rectifier.K.shape == (3, 3)
    assert rectifier.D.shape == (4, 1)
    assert rectifier.dim == (1920, 1080)
    assert rectifier.map1 is not None
    assert rectifier.map2 is not None

    # Test remapping a 1080p synthetic frame
    synthetic_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Draw a line from top-left to bottom-right
    synthetic_frame[100:200, 100:200] = (0, 255, 0)

    rectified = rectifier.rectify(synthetic_frame)
    assert rectified.shape == (1080, 1920, 3)
    assert rectified.dtype == np.uint8
    assert rectifier.last_rectify_ms >= 0.0
    print(f"\n[Test] Rectification 1080p execution latency: {rectifier.last_rectify_ms:.2f} ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
