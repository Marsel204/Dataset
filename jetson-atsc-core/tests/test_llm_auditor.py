"""
Unit tests for Asynchronous LLM Quality Auditor and Heuristic Critic.
"""

import os
import sys
import time
import pytest

CORE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)

from src.audit.llm_auditor import LLMAuditor


def test_llm_auditor_heuristic_anomaly_detection():
    auditor = LLMAuditor()

    # 1. Nominal Cycle: Parameters within balanced flow
    nominal_cycle = {
        "cycle_id": 1,
        "mode": "DUAL_CAM",
        "V_w": 25.0,
        "Q": 30.0,
        "L": 20.0,
        "t_anfis": 45.0,
        "n_served": 18,
        "fps": 28.5,
        "soc_temp_c": 52.0,
    }
    critique_nom = auditor._heuristic_critic(nominal_cycle)
    assert critique_nom["anomaly_detected"] is False
    assert critique_nom["severity"] == "NORMAL"

    # 2. Clamping Saturation Anomaly: Min green clamp (10s) despite heavy flow (40 PCU)
    clamp_cycle = {
        "cycle_id": 2,
        "mode": "DUAL_CAM",
        "V_w": 40.0,
        "Q": 35.0,
        "L": 35.0,
        "t_anfis": 10.0,  # Clamped to min!
        "n_served": 5,
        "fps": 28.0,
        "soc_temp_c": 53.0,
    }
    critique_clamp = auditor._heuristic_critic(clamp_cycle)
    assert critique_clamp["anomaly_detected"] is True
    assert "CLAMPING_SATURATION" in critique_clamp["anomaly_types"]

    # 3. Capacity Mismatch Anomaly: Long green (60s) but near-zero discharged vehicles (1)
    starvation_cycle = {
        "cycle_id": 3,
        "mode": "DUAL_CAM",
        "V_w": 35.0,
        "Q": 30.0,
        "L": 25.0,
        "t_anfis": 60.0,
        "n_served": 1,  # Starvation / occlusion
        "fps": 27.0,
        "soc_temp_c": 54.0,
    }
    critique_starv = auditor._heuristic_critic(starvation_cycle)
    assert critique_starv["anomaly_detected"] is True
    assert "CAPACITY_MISMATCH" in critique_starv["anomaly_types"]

    # 4. Thermal Degradation Anomaly: SoC temp = 78C
    thermal_cycle = {
        "cycle_id": 4,
        "mode": "DUAL_CAM",
        "V_w": 20.0,
        "Q": 15.0,
        "L": 15.0,
        "t_anfis": 30.0,
        "n_served": 12,
        "fps": 25.0,
        "soc_temp_c": 78.5,  # Overheating
    }
    critique_therm = auditor._heuristic_critic(thermal_cycle)
    assert critique_therm["anomaly_detected"] is True
    assert critique_therm["severity"] == "CRITICAL"
    assert "THERMAL_DEGRADATION" in critique_therm["anomaly_types"]


def test_llm_auditor_async_queue():
    completed_audits = []

    def on_complete(result):
        completed_audits.append(result)

    auditor = LLMAuditor(on_audit_completed=on_complete)
    auditor.start()

    sample_cycle = {
        "cycle_id": 10,
        "mode": "SINGLE_CAM",
        "V_w": 18.0,
        "Q": 20.0,
        "L": 15.0,
        "t_anfis": 32.0,
        "n_served": 10,
        "fps": 29.0,
        "soc_temp_c": 49.0,
    }

    auditor.enqueue_cycle(sample_cycle)
    t_start = time.time()
    while time.time() - t_start < 5.0 and len(completed_audits) == 0:
        time.sleep(0.05)
    auditor.stop()

    assert len(completed_audits) == 1
    assert completed_audits[0]["cycle_id"] == 10
    assert "audit" in completed_audits[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
