"""Measure motor response between live frames, independently of the target."""
from __future__ import annotations

from .engine import MeasurementEngine
from .models import FramePacket, MatchConfig, MatchResult


class LiveCalibration:
    """Own a stopped live baseline and measure subsequent native-pixel motion."""

    def __init__(self) -> None:
        self._engine = MeasurementEngine()
        self._config = MatchConfig(method="phase", scale=1.0)
        self._last_result: MatchResult | None = None

    def capture(self, frame: FramePacket) -> None:
        """Replace the baseline with an owned copy of this pre-move image."""
        self._engine.set_reference(frame)
        self._last_result = None

    def measure(self, frame: FramePacket) -> MatchResult:
        """Compare this image with the live baseline; right/down are positive."""
        if self._last_result is None or self._last_result.frame_id != frame.frame_id:
            self._last_result = self._engine.measure(frame, self._config)
        return self._last_result
