"""
Unit tests for domain types and dataclasses in src/common/types.py.
"""

import json
import os
import sys

CORE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)

from src.common.types import (
    TrafficSnapshot,
    ActuationDecision,
    HardwareTelemetry,
    CycleEvent,
)


def test_traffic_snapshot_from_dict_and_serialization():
    d = {
        "V_w": 18.4,
        "Q": 24.5,
        "L": 42.0,
        "n_mc": 12,
        "n_lv": 8,
        "n_hv": 2,
        "total_corridor_vehicles": 22,
        "stopped_vehicles_count": 9,
        "occupied_area_m2": 198.5,
    }
    snap = TrafficSnapshot.from_dict(d, timestamp=100.0)

    assert snap.v_w_pcu == 18.4
    assert snap.queue_meters == 24.5
    assert snap.occupancy_pct == 42.0
    assert snap.n_mc == 12
    assert snap.n_lv == 8
    assert snap.n_hv == 2
    assert snap.total_vehicles == 22
    assert snap.stopped_vehicles == 9
    assert snap.occupied_area_m2 == 198.5

    as_dict = snap.to_dict()
    assert as_dict["v_w_pcu"] == 18.4
    assert as_dict["queue_meters"] == 24.5


def test_actuation_decision_serialization():
    dec = ActuationDecision(
        cycle_id=5,
        green_seconds=34.5,
        min_bound=10.0,
        max_bound=120.0,
        is_clamped_min=False,
        is_clamped_max=False,
        inference_latency_ms=0.32,
        failsafe_active=False,
        timestamp_iso="2026-09-15T22:00:00Z",
    )
    d = dec.to_dict()
    assert d["cycle_id"] == 5
    assert d["green_seconds"] == 34.5
    assert d["failsafe_active"] is False


def test_cycle_event_ledger_row_schema():
    snap = TrafficSnapshot(v_w_pcu=12.0, queue_meters=15.0, occupancy_pct=30.0)
    dec = ActuationDecision(cycle_id=1, green_seconds=25.0)
    hw = HardwareTelemetry(soc_temp_c=45.2, fps=28.5, mode="DUAL_CAM")

    event = CycleEvent(
        cycle_id=1,
        timestamp_iso="2026-09-15T22:00:00Z",
        traffic=snap,
        decision=dec,
        telemetry=hw,
        n_served=10,
        theoretical_capacity_pcu=12.5,
    )

    row = event.to_ledger_row()
    assert row["cycle_id"] == 1
    assert row["v_w_pcu"] == 12.0
    assert row["q_meters"] == 15.0
    assert row["l_occupancy_pct"] == 30.0
    assert row["t_anfis_sec"] == 25.0
    assert row["n_served_actual"] == 10
    assert row["theoretical_capacity_pcu"] == 12.5
    assert row["fps"] == 28.5
    assert row["soc_temp_c"] == 45.2
    assert row["mode"] == "DUAL_CAM"

    # Verify JSON serialization works cleanly
    json_str = event.to_json()
    parsed = json.loads(json_str)
    assert parsed["cycle_id"] == 1
    assert parsed["traffic"]["v_w_pcu"] == 12.0
