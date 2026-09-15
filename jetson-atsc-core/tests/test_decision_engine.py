"""
Unit tests for pure ATSCDecisionEngine.
Verifies control boundary conditions, ANFIS inference isolation, and failsafe execution.
"""

import os
import sys
import pytest

CORE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)

from src.common.types import TrafficSnapshot, ActuationDecision
from src.control.anfis_inference import ANFISInferenceEngine
from src.control.decision_engine import ATSCDecisionEngine


@pytest.fixture
def decision_engine():
    model_path = os.path.join(CORE_ROOT, "models/anfis_sugeno.onnx")
    anfis = ANFISInferenceEngine(model_path, prefer_cuda=False, min_green_sec=10.0, max_green_sec=120.0)
    return ATSCDecisionEngine(anfis, min_green_sec=10.0, max_green_sec=120.0, fallback_green_sec=30.0)


def test_decision_engine_empty_intersection(decision_engine):
    snap = TrafficSnapshot(v_w_pcu=0.0, queue_meters=0.0, occupancy_pct=0.0)
    dec = decision_engine.evaluate(snap, cycle_id=1)

    assert isinstance(dec, ActuationDecision)
    assert dec.cycle_id == 1
    assert dec.green_seconds >= 10.0
    assert dec.green_seconds <= 120.0
    assert dec.failsafe_active is False
    assert dec.inference_latency_ms > 0.0


def test_decision_engine_high_demand(decision_engine):
    snap = TrafficSnapshot(v_w_pcu=60.0, queue_meters=45.0, occupancy_pct=95.0)
    dec = decision_engine.evaluate(snap, cycle_id=2)

    assert dec.cycle_id == 2
    assert dec.green_seconds > 40.0
    assert dec.green_seconds <= 120.0
    assert dec.failsafe_active is False


def test_decision_engine_fallback():
    class BrokenANFIS:
        def evaluate_green_time(self, *args, **kwargs):
            raise RuntimeError("CUDA out of memory simulation")

    engine = ATSCDecisionEngine(BrokenANFIS(), min_green_sec=10.0, max_green_sec=120.0, fallback_green_sec=30.0)
    snap = TrafficSnapshot(v_w_pcu=10.0, queue_meters=10.0, occupancy_pct=20.0)
    dec = engine.evaluate(snap, cycle_id=99)

    assert dec.cycle_id == 99
    assert dec.green_seconds == 30.0
    assert dec.failsafe_active is True
