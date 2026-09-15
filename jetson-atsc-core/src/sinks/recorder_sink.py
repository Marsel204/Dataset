"""
Async Video Recorder Sink.

Enqueues ground-truth video frames with stamped telemetry watermarks,
keeping disk I/O isolated from the real-time perception loop.
"""

from __future__ import annotations

import os
from typing import Optional
import numpy as np

from src.common.types import CycleEvent
from src.sinks.base import CycleSink
from src.storage.async_recorder import AsyncRecorder


class RecorderSink(CycleSink):
    """
    Adapter sink integrating the non-blocking MP4 AsyncRecorder into the observer pipeline.
    """

    def __init__(self, output_dir: str, filename_prefix: str = "kamera_akuisisi") -> None:
        self.output_dir = os.path.abspath(output_dir)
        self.recorder = AsyncRecorder(output_dir=self.output_dir, filename_prefix=filename_prefix)

    def start(self) -> None:
        self.recorder.start()

    def enqueue_frame(self, frame: np.ndarray, watermark: str = "") -> None:
        """Enqueues video frame with optional watermark."""
        self.recorder.enqueue_frame(frame, watermark)

    def on_cycle(self, event: CycleEvent) -> None:
        """Optional hook for cycle events (e.g. tagging video landmarks)."""
        pass

    def close(self) -> None:
        self.recorder.stop()
