"""
Base abstraction for ATSC cycle event sinks.

Allows observers (CSV ledgers, audit workers, cloud sync, telemetry recorders)
to be cleanly attached to the control cycle without coupling to the core daemon loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from src.common.types import CycleEvent


class CycleSink(ABC):
    """
    Abstract observer interface for handling completed traffic actuation cycles.
    """

    @abstractmethod
    def on_cycle(self, event: CycleEvent) -> None:
        """Invoked immediately after a traffic actuation decision is finalized."""
        pass

    def start(self) -> None:
        """Optional lifecycle hook to start background threads or workers."""
        pass

    def close(self) -> None:
        """Optional lifecycle hook to flush buffers and cleanly release resources."""
        pass
