"""
CSV Experiment Ledger Sink.

Appends structured CycleEvents to field_experiment_ledger.csv and updates
asynchronous LLM audit critiques when analysis completes.
"""

from __future__ import annotations

import csv
import os
import threading
from typing import Any, Dict, List, Optional

from src.common.types import CycleEvent
from src.sinks.base import CycleSink


class LedgerSink(CycleSink):
    """
    Persists real-time cycle results to a structured CSV ledger.
    """

    def __init__(self, ledger_path: str) -> None:
        self.ledger_path = os.path.abspath(ledger_path)
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)

    def on_cycle(self, event: CycleEvent) -> None:
        """Appends a cycle record to the CSV file."""
        row = event.to_ledger_row()
        with self._lock:
            file_exists = os.path.exists(self.ledger_path)
            is_empty = not file_exists or os.path.getsize(self.ledger_path) == 0

            with open(self.ledger_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if is_empty:
                    writer.writeheader()
                writer.writerow(row)

    def update_audit_result(self, cycle_id: int | str, audit_dict: Dict[str, Any]) -> None:
        """Updates the ledger with asynchronous LLM audit verdicts."""
        with self._lock:
            if not os.path.exists(self.ledger_path):
                return

            rows: List[Dict[str, Any]] = []
            fieldnames = None
            try:
                with open(self.ledger_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames
                    for r in reader:
                        if str(r.get("cycle_id")) == str(cycle_id):
                            r["llm_anomaly_flag"] = str(audit_dict.get("anomaly_detected", False))
                            r["llm_severity"] = audit_dict.get("severity", "NORMAL")
                            r["llm_critique"] = str(audit_dict.get("critique", "")).replace(",", ";")
                        rows.append(r)

                if fieldnames and rows:
                    with open(self.ledger_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)
            except Exception as e:
                print(f"[LedgerSink] Error updating audit in ledger: {e}")
