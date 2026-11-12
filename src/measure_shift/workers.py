"""Bounded background work. No worker accesses a Qt widget."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
import threading
import time
from typing import Any, Callable

from .models import FramePacket, InspectionPair, MatchConfig, ReferenceSnapshot


class LatestValue:
    """A single replaceable value, with independent cursors for each consumer."""

    def __init__(self):
        self._condition = threading.Condition()
        self._value: Any = None
        self._revision = 0

    def publish(self, value: Any) -> None:
        with self._condition:
            self._value = value
            self._revision += 1
            self._condition.notify_all()

    def read(self) -> tuple[Any, int]:
        with self._condition:
            return self._value, self._revision

    def wait(self, revision: int, timeout: float = 0.05) -> tuple[Any, int]:
        with self._condition:
            self._condition.wait_for(lambda: self._revision != revision, timeout)
            return self._value, self._revision

    def clear(self) -> None:
        self.publish(None)


class RateMeter:
    def __init__(self):
        self._times: deque[float] = deque(maxlen=120)
        self._value = 0.0
        self._last_at = 0.0

    @property
    def value(self) -> float:
        if time.monotonic() - self._last_at > max(1.0, 3 / max(.1, self._value)):
            return 0.0
        return self._value

    def tick(self) -> None:
        now = time.monotonic()
        self._last_at = now
        self._times.append(now)
        while len(self._times) > 2 and now - self._times[0] > 3:
            self._times.popleft()
        if len(self._times) > 1:
            self._value = (len(self._times) - 1) / max(now - self._times[0], 1e-9)


class CaptureWorker(threading.Thread):
    def __init__(self, source, mailbox: LatestValue, *, maximize_timing=False):
        super().__init__(name="camera-acquisition", daemon=True)
        self.source = source
        self.mailbox = mailbox
        self.stopping = threading.Event()
        self.status = "Connecting"
        self.error: str | None = None
        self.rate = RateMeter()
        self.controls: dict = {}
        self._parameters: dict = {}
        self._lock = threading.Lock()
        self._publication_lock = threading.Lock()
        self._freshness_generation = 0
        self._freshness_after = 0.0
        self.parameter_error: str | None = None
        self.maximize_timing = maximize_timing

    def _maximize_timing(self) -> None:
        errors = []
        # Each setting changes the next setting's limits. Read the camera again
        # between writes, and finish before publishing controls or any frames.
        for key in ("pixel_clock_mhz", "fps", "exposure_ms"):
            if self.stopping.is_set():
                break
            try:
                info = self.source.controls().get(key)
                if info is None:
                    continue
                value = max(info["values"]) if info.get("values") else info["max"]
                if value != info["value"]:
                    self.source.set_parameters(**{key: value})
            except Exception as exc:
                errors.append(f"Startup {key}: {exc}")
        self.parameter_error = "; ".join(errors) or None

    def set_parameters(self, **kwargs) -> None:
        with self._lock:
            self._parameters.update(kwargs)

    def request_fresh_frames(self, after: float) -> int:
        """Return a barrier token for frames acquired after a settling cutoff.

        Queue clearing happens on the acquisition thread. An in-flight read or
        ownership copy cannot subsequently publish as part of the new generation.
        Consumers must check both the generation and ``read_started_at``; receipt
        or measurement completion alone cannot establish post-motion freshness.
        """
        after = float(after)
        if not math.isfinite(after):
            raise ValueError("Fresh-frame cutoff must be finite")
        with self._publication_lock:
            with self._lock:
                self._freshness_generation += 1
                self._freshness_after = after
                return self._freshness_generation

    def run(self) -> None:
        try:
            self.source.open()
            if self.maximize_timing:
                self._maximize_timing()
            self.controls = self.source.controls()
            self.status = "Connected"
            freshness_generation = 0
            discard_first_fresh_frame = False
            while not self.stopping.is_set():
                with self._lock:
                    params, self._parameters = self._parameters, {}
                    requested_generation = self._freshness_generation
                    freshness_after = self._freshness_after
                if params:
                    try:
                        self.source.set_parameters(**params)
                        self.controls = self.source.controls()
                        self.parameter_error = None
                    except Exception as exc:
                        self.parameter_error = str(exc)
                        try:
                            self.controls = self.source.controls()
                        except Exception:
                            pass
                if (requested_generation != freshness_generation
                        and time.monotonic() >= freshness_after):
                    # Clearing/restarting the source discards frames buffered
                    # before the cutoff, including the legacy IDS three-buffer
                    # queue. Keep all SDK operations on this thread.
                    discard_pending = getattr(self.source, "discard_pending_frames", None)
                    if discard_pending is not None:
                        discard_pending()
                    freshness_generation = requested_generation
                    # A first exposure may have started before a free-running
                    # camera's queue was cleared. Never use that frame as feedback.
                    discard_first_fresh_frame = True
                read_started_at = time.monotonic()
                read_started = time.perf_counter()
                pixels = self.source.read(timeout_ms=200)
                read_ms = (time.perf_counter() - read_started) * 1000
                if pixels is None:
                    if getattr(self.source, "ended", False):
                        self.status = "End of input"
                        break
                    self.stopping.wait(0.005)
                    continue
                if self.stopping.is_set():
                    break
                if discard_first_fresh_frame:
                    discard_first_fresh_frame = False
                    continue
                packet = replace(FramePacket.from_array(pixels, self.source.name), source_wait_read_ms=read_ms,
                                 read_started_at=read_started_at, freshness_generation=freshness_generation)
                with self._publication_lock:
                    if self.stopping.is_set():
                        break
                    with self._lock:
                        if (freshness_generation != self._freshness_generation
                                or read_started_at < self._freshness_after):
                            continue
                    self.mailbox.publish(packet)
                self.rate.tick()
        except Exception as exc:
            self.error = str(exc)
            self.status = "Disconnected"
        finally:
            try:
                self.source.close()
            except Exception as exc:
                self.error = self.error or str(exc)
            if self.stopping.is_set():
                self.status = "Disconnected"

    def stop(self) -> None:
        with self._publication_lock:
            self.stopping.set()


class MeasurementWorker(threading.Thread):
    def __init__(self, frames: LatestValue):
        super().__init__(name="alignment-measurement", daemon=True)
        self.frames = frames
        self.output = LatestValue()
        self.stopping = threading.Event()
        self.paused = threading.Event()
        self.rate = RateMeter()
        self.error: str | None = None
        self._lock = threading.Lock()
        self._reference: ReferenceSnapshot | None = None
        self._config = MatchConfig()
        self.calibration_output = LatestValue()
        self._calibration_request: tuple[float, int] | None = None

    def start_calibration(self, after: float, generation: int) -> None:
        """Select a fresh live baseline without replacing the saved reference."""
        with self._lock:
            self._calibration_request = after, generation
            self.calibration_output.clear()

    def stop_calibration(self) -> None:
        with self._lock:
            self._calibration_request = None
            self.calibration_output.clear()

    def configure(self, reference: ReferenceSnapshot | None, config: MatchConfig) -> None:
        with self._lock:
            self._reference, self._config = reference, config

    def current_key(self) -> tuple[str | None, str]:
        with self._lock:
            return self._reference.reference_id if self._reference else None, self._config.config_id

    def run(self) -> None:
        from .engine import MeasurementEngine
        from .live_calibration import CalibrationObservation, LiveCalibration

        engine = MeasurementEngine()
        revision = -1
        previous_key = None
        calibration = None
        calibration_request = None
        while not self.stopping.is_set():
            frame, revision = self.frames.wait(revision)
            if self.paused.is_set() or frame is None:
                continue
            with self._lock:
                reference, config = self._reference, self._config
                request = self._calibration_request
            if request != calibration_request:
                calibration = None
                calibration_request = request
            # Calibration must also run when no saved reference exists. Its
            # own engine keeps target matching regions out of the motor
            # response measurement.
            if request is not None:
                after, generation = request
                if (frame.freshness_generation >= generation and frame.read_started_at is not None
                        and frame.read_started_at >= after and frame.received_at >= after):
                    try:
                        if calibration is None:
                            calibration = LiveCalibration()
                            calibration.capture(frame)
                        observation = CalibrationObservation(request, frame, calibration.measure(frame))
                        with self._lock:
                            if request == self._calibration_request:
                                self.calibration_output.publish(observation)
                    except Exception as exc:
                        self.error = str(exc)
            if reference is None:
                continue
            key = (frame.frame_id, reference.reference_id, config.config_id)
            if key == previous_key:
                continue
            previous_key = key
            try:
                if engine.reference is None or engine.reference.reference_id != reference.reference_id:
                    engine.set_reference(reference)
                result = engine.measure(frame, config)
                # A slow old job must never replace results for a changed reference/ROI.
                if self.current_key() == (reference.reference_id, config.config_id):
                    self.output.publish(InspectionPair(reference, frame, config, result))
                    self.rate.tick()
                    self.error = None
            except Exception as exc:
                self.error = str(exc)

    def stop(self) -> None:
        self.stopping.set()


@dataclass(frozen=True)
class AnalysisOutput:
    generation: int
    snapshot: Any = None
    error: str | None = None


class AnalyticsWorker(threading.Thread):
    """One executing job and one replaceable request; cancellation drops results."""

    def __init__(self, analyzer: Callable | None = None):
        super().__init__(name="optional-analytics", daemon=True)
        self.output = LatestValue()
        self.stopping = threading.Event()
        self._condition = threading.Condition()
        self._pending: tuple | None = None
        self._generation = 0
        self._analyzer = analyzer
        self.jobs_completed = 0
        self.busy = False

    @property
    def generation(self) -> int:
        with self._condition:
            return self._generation

    def cancel(self) -> int:
        with self._condition:
            self._generation += 1
            self._pending = None
            self.output.clear()
            return self._generation

    def submit(self, pair: InspectionPair, tab: str) -> int:
        with self._condition:
            self._pending = (self._generation, pair, tab)
            self._condition.notify_all()
            return self._generation

    def run(self) -> None:
        if self._analyzer is None:
            from .analytics import analyze_pair
            self._analyzer = analyze_pair
        while not self.stopping.is_set():
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self.stopping.is_set())
                if self.stopping.is_set():
                    break
                generation, pair, tab = self._pending
                self._pending = None
                self.busy = True
            try:
                snapshot = self._analyzer(pair, tab)
                output = AnalysisOutput(generation, snapshot=snapshot)
            except Exception as exc:
                output = AnalysisOutput(generation, error=str(exc))
            finally:
                self.busy = False
            with self._condition:
                if generation == self._generation and not self.stopping.is_set():
                    self.output.publish(output)
                    self.jobs_completed += 1
            # The idle thread must not retain the last large diagnostic payload
            # after cancellation clears its mailbox. In-flight work can finish,
            # but disabling analytics releases its arrays without another job.
            snapshot = None
            output = None
            pair = None

    def stop(self) -> None:
        with self._condition:
            self.stopping.set()
            self._pending = None
            self._condition.notify_all()


class BackgroundTask(threading.Thread):
    def __init__(self, function: Callable, callback: Callable, error_callback: Callable):
        super().__init__(name="application-task", daemon=True)
        self.function, self.callback, self.error_callback = function, callback, error_callback
        self.result = None
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            self.result = self.function()
        except Exception as exc:
            self.error = exc
