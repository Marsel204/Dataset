"""
Asynchronous LLM Quality Auditor Sink.

Inspects completed cycle events using an LLM or fallback rule engine
to flag physical traffic flow anomalies, clamping saturation, and thermal degradation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from src.common.types import CycleEvent
from src.sinks.base import CycleSink
from src.audit.llm_auditor import LLMAuditor


class LLMAuditSink(CycleSink):
    """
    Adapter sink forwarding cycle events to the asynchronous LLM quality auditor.
    """

    def __init__(
        self,
        on_audit_completed: Optional[Callable[[Dict[str, Any]], None]] = None,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
    ) -> None:
        self.auditor = LLMAuditor(
            api_key=api_key,
            base_url=base_url,
            model=model,
            on_audit_completed=on_audit_completed,
        )

    def start(self) -> None:
        self.auditor.start()

    def on_cycle(self, event: CycleEvent) -> None:
        """Extracts cycle audit payload and submits to background audit queue."""
        payload = {
            "cycle_id": event.cycle_id,
            "timestamp": event.timestamp_iso,
            "mode": event.telemetry.mode,
            "V_w": event.traffic.v_w_pcu,
            "Q": event.traffic.queue_meters,
            "L": event.traffic.occupancy_pct,
            "t_anfis": event.decision.green_seconds,
            "n_served": event.n_served,
            "fps": event.telemetry.fps,
            "soc_temp_c": event.telemetry.soc_temp_c,
        }
        self.auditor.enqueue_cycle(payload)

    def close(self) -> None:
        self.auditor.stop()
