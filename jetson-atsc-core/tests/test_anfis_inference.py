"""
Unit tests for 126-Parameter Sugeno ANFIS ONNX Inference Engine.
"""

import os
import sys
import numpy as np
import pytest

CORE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)

from src.control.anfis_inference import ANFISInferenceEngine


def test_anfis_inference_forward_pass_and_bounds():
    model_path = os.path.join(CORE_ROOT, "models/anfis_sugeno.onnx")
    assert os.path.exists(model_path), f"ONNX model missing at: {model_path}"

    engine = ANFISInferenceEngine(
        model_path=model_path,
        prefer_cuda=True,
        min_green_sec=10.0,
        max_green_sec=120.0,
    )

    # 1. Evaluate baseline moderate traffic
    t_green = engine.evaluate_green_time(V_w=25.0, Q=35.0, L=20.0)
    assert 10.0 <= t_green <= 120.0
    assert engine.last_inference_ms < 50.0  # Sub-50ms constraint
    print(f"\n[Test] Moderate Traffic: t_ANFIS = {t_green:.2f}s (Latency: {engine.last_inference_ms:.3f}ms)")

    # 2. Test extreme low traffic: Should be bounded >= 10.0s
    t_min = engine.evaluate_green_time(V_w=0.0, Q=0.0, L=0.0)
    assert t_min >= 10.0
    print(f"[Test] Minimal Traffic: t_ANFIS = {t_min:.2f}s")

    # 3. Test severe congestion: Should be bounded <= 120.0s
    t_max = engine.evaluate_green_time(V_w=100.0, Q=200.0, L=95.0)
    assert t_max <= 120.0
    print(f"[Test] Severe Congestion: t_ANFIS = {t_max:.2f}s")

    # 4. Test deterministic monotonicity: High traffic should receive more green than low traffic
    assert t_max >= t_min


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
