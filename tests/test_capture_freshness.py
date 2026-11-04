"""Post-motion image barriers without cameras or motor hardware."""
from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Queue
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from measure_shift.models import FramePacket
from measure_shift.sources import BaslerSource, IDSSource, SourceError
from measure_shift.workers import CaptureWorker, LatestValue


@dataclass
class ReadGate:
    values: Queue = field(default_factory=Queue)

    def finish(self, value):
        self.values.put(np.full((8, 12), value, np.uint8))


class GatedCamera:
    name = "Test camera"
    ended = False

    def __init__(self):
        self.reads = Queue()
        self.flushes = []
        self.closed = False

    def open(self):
        pass

    def controls(self):
        return {}

    def read(self, timeout_ms=200):
        gate = ReadGate()
        self.reads.put(gate)
        try:
            return gate.values.get(timeout=timeout_ms / 1000)
        except Empty:
            return None

    def discard_pending_frames(self):
        self.flushes.append((time.monotonic(), threading.get_ident()))

    def close(self):
        self.closed = True


def finish(worker):
    worker.stop()
    worker.join(timeout=2)
    assert not worker.is_alive()


def test_barrier_rejects_inflight_read_and_first_restarted_frame():
    source, frames = GatedCamera(), LatestValue()
    worker = CaptureWorker(source, frames)
    worker.start()
    try:
        inflight = source.reads.get(timeout=2)
        cutoff = time.monotonic()
        token = worker.request_fresh_frames(cutoff)
        inflight.finish(11)
        first_restarted = source.reads.get(timeout=2)
        assert len(source.flushes) == 1
        assert source.flushes[0][0] >= cutoff
        assert source.flushes[0][1] == worker.ident
        assert frames.read() == (None, 0)
        first_restarted.finish(22)
        fresh = source.reads.get(timeout=2)
        assert frames.read() == (None, 0)
        fresh.finish(33)
        packet, revision = frames.wait(0, timeout=2)
        assert revision == 1
        assert packet.freshness_generation == token
        assert cutoff <= packet.read_started_at <= packet.received_at
        np.testing.assert_array_equal(packet.pixels, 33)
    finally:
        finish(worker)
    assert source.closed


def test_newer_barrier_replaces_pending_generation():
    source, frames = GatedCamera(), LatestValue()
    worker = CaptureWorker(source, frames)
    worker.start()
    try:
        inflight = source.reads.get(timeout=2)
        first = worker.request_fresh_frames(time.monotonic())
        second = worker.request_fresh_frames(time.monotonic())
        assert second == first + 1
        inflight.finish(11)
        source.reads.get(timeout=2).finish(22)
        source.reads.get(timeout=2).finish(33)
        packet, _ = frames.wait(0, timeout=2)
        assert packet.freshness_generation == second
        assert len(source.flushes) == 1
    finally:
        finish(worker)


def test_barrier_requested_during_frame_copy_cannot_publish_old_frame(monkeypatch):
    source, frames = GatedCamera(), LatestValue()
    worker = CaptureWorker(source, frames)
    copy_started, release_copy = threading.Event(), threading.Event()
    original = FramePacket.from_array
    calls = 0

    def slow_copy(cls, pixels, source_name="Image"):
        nonlocal calls
        calls += 1
        if calls == 1:
            copy_started.set()
            assert release_copy.wait(2)
        return original(pixels, source_name)

    monkeypatch.setattr(FramePacket, "from_array", classmethod(slow_copy))
    worker.start()
    try:
        source.reads.get(timeout=2).finish(11)
        assert copy_started.wait(2)
        token = worker.request_fresh_frames(time.monotonic())
        release_copy.set()
        source.reads.get(timeout=2).finish(22)
        assert frames.read() == (None, 0)
        source.reads.get(timeout=2).finish(33)
        packet, _ = frames.wait(0, timeout=2)
        assert packet.freshness_generation == token
        np.testing.assert_array_equal(packet.pixels, 33)
    finally:
        release_copy.set()
        finish(worker)


def test_queue_clear_waits_for_settling_cutoff():
    source, frames = GatedCamera(), LatestValue()
    worker = CaptureWorker(source, frames)
    worker.start()
    try:
        gate = source.reads.get(timeout=2)
        cutoff = time.monotonic() + 0.05
        token = worker.request_fresh_frames(cutoff)
        gate.finish(11)
        gate = source.reads.get(timeout=2)
        assert not source.flushes
        assert frames.read() == (None, 0)
        # This read started before settling completed and cannot be published.
        remaining = cutoff - time.monotonic()
        if remaining > 0:
            threading.Event().wait(remaining)
        gate.finish(22)
        source.reads.get(timeout=2).finish(33)  # first restarted frame is discarded
        source.reads.get(timeout=2).finish(44)
        packet, _ = frames.wait(0, timeout=2)
        assert source.flushes[0][0] >= cutoff
        assert packet.read_started_at >= cutoff
        assert packet.freshness_generation == token
        np.testing.assert_array_equal(packet.pixels, 44)
    finally:
        finish(worker)


@pytest.mark.parametrize("after", [float("nan"), float("inf"), -float("inf")])
def test_barrier_rejects_nonfinite_cutoffs(after):
    worker = CaptureWorker(GatedCamera(), LatestValue())
    with pytest.raises(ValueError, match="finite"):
        worker.request_fresh_frames(after)


def test_basler_barrier_restarts_latest_only_acquisition():
    calls = []
    source = BaslerSource()
    source._camera = SimpleNamespace(
        StopGrabbing=lambda: calls.append("stop"),
        StartGrabbing=lambda strategy: calls.append(("start", strategy)),
    )
    source._pylon = SimpleNamespace(GrabStrategy_LatestImageOnly=73)
    source.discard_pending_frames()
    assert calls == ["stop", ("start", 73)]


@pytest.mark.parametrize("modern", [False, True])
def test_ids_barrier_stops_clears_queue_and_restarts_both_sdk_variants(modern):
    calls = []

    def record(name):
        def call(*args):
            calls.append(name)
            return 0
        return call

    source = IDSSource()
    source._handle = object()
    source._capturing = source._queue_started = True
    source._modern_queue = modern
    source._ueye = SimpleNamespace(
        IS_SUCCESS=0, IS_FORCE_VIDEO_STOP=4, IS_DONT_WAIT=5,
        IS_IMAGE_QUEUE_CMD_EXIT=10, IS_IMAGE_QUEUE_CMD_INIT=11,
        is_StopLiveVideo=record("stop"), is_CaptureVideo=record("start"),
        is_ImageQueue=lambda handle, command, *_: record("exit" if command == 10 else "init")(),
        is_ExitImageQueue=record("exit"), is_InitImageQueue=record("init"),
    )
    source.discard_pending_frames()
    assert calls == ["stop", "exit", "init", "start"]
    assert source._capturing and source._queue_started


@pytest.mark.parametrize("source", [BaslerSource(), IDSSource()])
def test_unopened_camera_cannot_satisfy_barrier(source):
    with pytest.raises(SourceError):
        source.discard_pending_frames()
