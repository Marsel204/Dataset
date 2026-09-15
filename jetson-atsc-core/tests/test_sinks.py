"""
Unit tests for pluggable CycleSink implementations in src/sinks/.
"""

import csv
import os
import sys
from src.common.types import (
    TrafficSnapshot,
    ActuationDecision,
    HardwareTelemetry,
    CycleEvent,
)
from src.sinks.ledger_sink import LedgerSink


def test_ledger_sink_writes_and_updates(tmp_path):
    ledger_file = str(tmp_path / "test_ledger.csv")
    sink = LedgerSink(ledger_file)

    event = CycleEvent(
        cycle_id=1,
        timestamp_iso="2026-09-15T22:00:00Z",
        traffic=TrafficSnapshot(v_w_pcu=14.0, queue_meters=20.0, occupancy_pct=35.0),
        decision=ActuationDecision(cycle_id=1, green_seconds=28.5),
        telemetry=HardwareTelemetry(soc_temp_c=50.0, fps=25.0, mode="DUAL_CAM"),
        n_served=8,
        theoretical_capacity_pcu=14.2,
    )

    # 1. On cycle
    sink.on_cycle(event)
    assert os.path.exists(ledger_file)

    with open(ledger_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["cycle_id"] == "1"
    assert rows[0]["llm_anomaly_flag"] == "PENDING"

    # 2. Update audit result
    sink.update_audit_result(1, {
        "anomaly_detected": True,
        "severity": "MEDIUM",
        "critique": "Minor queue buildup detected.",
    })

    with open(ledger_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["llm_anomaly_flag"] == "True"
    assert rows[0]["llm_severity"] == "MEDIUM"
    assert "Minor queue buildup" in rows[0]["llm_critique"]
