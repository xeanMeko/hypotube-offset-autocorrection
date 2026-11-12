"""Deterministic background scheduling tests without importing Qt."""
from dataclasses import FrozenInstanceError, replace
import gc
import threading
import time
import weakref

import numpy as np
import pytest

from measure_shift.models import DiagnosticSnapshot, FramePacket, InspectionPair, MatchConfig, MatchResult, ReferenceSnapshot
from measure_shift.workers import AnalyticsWorker, LatestValue, MeasurementWorker


def make_pair():
    frame = FramePacket.from_array(np.arange(32 * 48, dtype=np.uint8).reshape(32, 48))
    reference = ReferenceSnapshot(FramePacket.from_array(frame.pixels))
    config = MatchConfig()
    result = MatchResult(frame.frame_id, reference.reference_id, config.config_id, dx=0, dy=0, validity="valid")
    return InspectionPair(reference, frame, config, result)


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("Background worker did not reach the expected state")
        threading.Event().wait(0.005)


def stop(worker):
    worker.stop()
    worker.join(timeout=2)
    assert not worker.is_alive()


def test_latest_value_retains_one_value_and_independent_revision_cursors():
    mailbox = LatestValue()
    assert mailbox.read() == (None, 0)
    for value in range(1000):
        mailbox.publish(value)
    assert mailbox.read() == (999, 1000)
    # Reading/waiting is non-destructive for independent consumers.
    assert mailbox.wait(0, timeout=0) == (999, 1000)
    assert mailbox.wait(995, timeout=0) == (999, 1000)
    assert mailbox.wait(1000, timeout=0) == (999, 1000)
    mailbox.clear()
    assert mailbox.read() == (None, 1001)


@pytest.mark.parametrize("changed", ["reference", "config"])
def test_measurement_rejects_old_inflight_reference_or_configuration(monkeypatch, changed):
    import measure_shift.engine as engine_module

    started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    release_second = threading.Event()
    calls = []

    class SlowEngine:
        reference = None

        def set_reference(self, reference):
            self.reference = reference

        def measure(self, frame, config):
            calls.append((self.reference.reference_id, config.config_id))
            if len(calls) == 1:
                started.set()
                assert release_first.wait(2)
            else:
                second_started.set()
                assert release_second.wait(2)
            return MatchResult(frame.frame_id, self.reference.reference_id, config.config_id,
                               dx=0, dy=0, validity="valid")

    monkeypatch.setattr(engine_module, "MeasurementEngine", SlowEngine)
    pair = make_pair()
    next_pair = make_pair()
    reference = next_pair.reference if changed == "reference" else pair.reference
    config = next_pair.config if changed == "config" else pair.config
    frames = LatestValue()
    frames.publish(pair.frame)
    worker = MeasurementWorker(frames)
    worker.configure(pair.reference, pair.config)
    worker.start()
    try:
        assert started.wait(2)
        worker.configure(reference, config)
        release_first.set()
        assert second_started.wait(2)
        assert worker.output.read() == (None, 0)
        release_second.set()
        output, _ = worker.output.wait(0, timeout=2)
        assert output.reference is reference
        assert output.config is config
        assert output.result.reference_id == reference.reference_id
        assert output.result.config_id == config.config_id
    finally:
        release_first.set()
        release_second.set()
        stop(worker)


def test_live_calibration_uses_fresh_camera_frames_without_a_saved_reference():
    frames = LatestValue()
    worker = MeasurementWorker(frames)
    image = np.random.default_rng(12).integers(15, 240, (128, 160), dtype=np.uint8)

    def frame(pixels, *, started=10.01, received=10.02, generation=3):
        return replace(FramePacket.from_array(pixels), read_started_at=started,
                       received_at=received, freshness_generation=generation)

    worker.start_calibration(10, 3)
    worker.start()
    try:
        for stale in (frame(image, generation=2), frame(image, started=9),
                      frame(image, started=None), frame(image, received=9)):
            frames.publish(stale)
            assert worker.calibration_output.wait(worker.calibration_output.read()[1], .06)[0] is None

        baseline = frame(image)
        frames.publish(baseline)
        wait_for(lambda: worker.calibration_output.read()[0] is not None)
        before = worker.calibration_output.read()[0]
        assert before.frame is baseline
        assert before.result.usable
        assert before.result.dx == pytest.approx(0, abs=.1)

        after = frame(np.roll(image, -6, axis=1), started=11, received=11.01, generation=4)
        frames.publish(after)
        wait_for(lambda: worker.calibration_output.read()[0].frame is after)
        observation = worker.calibration_output.read()[0]
        assert observation.request == (10, 3)
        assert observation.result.usable
        assert observation.result.dx == pytest.approx(-6, abs=.3)
        assert worker.output.read()[0] is None

        worker.stop_calibration()
        assert worker.calibration_output.read()[0] is None
        worker.start_calibration(20, 5)
        baseline = frame(after.pixels, started=20.01, received=20.02, generation=5)
        frames.publish(baseline)
        wait_for(lambda: worker.calibration_output.read()[0] is not None)
        restarted = worker.calibration_output.read()[0]
        assert restarted.request == (20, 5)
        assert restarted.result.reference_id != before.result.reference_id
        assert restarted.result.dx == pytest.approx(0, abs=.1)
    finally:
        stop(worker)


@pytest.mark.parametrize("restart", [False, True])
def test_live_calibration_discards_inflight_results_after_cancellation(monkeypatch, restart):
    from measure_shift.live_calibration import LiveCalibration
    started, release = threading.Event(), threading.Event()
    measure = LiveCalibration.measure

    def slow_measure(self, frame):
        started.set()
        assert release.wait(2)
        return measure(self, frame)

    monkeypatch.setattr(LiveCalibration, "measure", slow_measure)
    frames = LatestValue()
    worker = MeasurementWorker(frames)
    worker.start_calibration(10, 1)
    worker.start()
    try:
        frames.publish(replace(make_pair().frame, read_started_at=10.01,
                               received_at=10.02, freshness_generation=1))
        assert started.wait(2)
        worker.stop_calibration()
        if restart:
            worker.start_calibration(20, 2)
        release.set()
        assert worker.calibration_output.wait(worker.calibration_output.read()[1], .1)[0] is None
    finally:
        release.set()
        stop(worker)


def test_analytics_performs_zero_jobs_until_submitted():
    called = threading.Event()

    def analyzer(pair, tab):
        called.set()
        return DiagnosticSnapshot(pair, tab)

    worker = AnalyticsWorker(analyzer)
    worker.start()
    try:
        assert not called.wait(0.03)
        assert worker.jobs_completed == 0
        assert worker.output.read() == (None, 0)
        pair = make_pair()
        generation = worker.submit(pair, "Inputs")
        output, _ = worker.output.wait(0, timeout=2)
        assert called.is_set()
        assert output.generation == generation
        assert output.snapshot.pair is pair
        assert output.error is None
        assert worker.jobs_completed == 1
    finally:
        stop(worker)


def test_analytics_cancel_drops_inflight_and_pending_then_accepts_new_generation():
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def analyzer(pair, tab):
        calls.append(tab)
        if tab == "first":
            entered.set()
            assert release.wait(2)
        return DiagnosticSnapshot(pair, tab)

    worker = AnalyticsWorker(analyzer)
    pair = make_pair()
    worker.start()
    try:
        worker.submit(pair, "first")
        assert entered.wait(2)
        worker.submit(pair, "cancelled pending")
        generation = worker.cancel()
        worker.submit(pair, "new")
        release.set()
        wait_for(lambda: worker.output.read()[0] is not None)
        output, _ = worker.output.read()
        assert output.generation == generation
        assert output.snapshot.tab == "new"
        assert calls == ["first", "new"]
        assert worker.jobs_completed == 1
        worker.cancel()
        assert worker.output.read()[0] is None
    finally:
        release.set()
        stop(worker)


def test_analytics_replaces_pending_request_instead_of_accumulating_work():
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def analyzer(pair, tab):
        calls.append(tab)
        if tab == "first":
            entered.set()
            assert release.wait(2)
        return DiagnosticSnapshot(pair, tab)

    worker = AnalyticsWorker(analyzer)
    pair = make_pair()
    worker.start()
    try:
        worker.submit(pair, "first")
        assert entered.wait(2)
        worker.submit(pair, "superseded")
        worker.submit(pair, "latest")
        release.set()
        wait_for(lambda: worker.jobs_completed == 2)
        assert calls == ["first", "latest"]
        assert worker.output.read()[0].snapshot.tab == "latest"
    finally:
        release.set()
        stop(worker)


def test_cancel_releases_idle_worker_diagnostic_arrays_without_another_job():
    array_refs = []

    def analyzer(pair, tab):
        array = np.zeros((256, 256), dtype=np.float32)
        array_refs.append(weakref.ref(array))
        return DiagnosticSnapshot(pair, tab, arrays={"large": array})

    worker = AnalyticsWorker(analyzer)
    pair = make_pair()
    reference_pixels = weakref.ref(pair.reference.frame.pixels)
    live_pixels = weakref.ref(pair.frame.pixels)
    worker.start()
    try:
        worker.submit(pair, "Inputs")
        output, _ = worker.output.wait(0, timeout=2)
        assert output.snapshot.pair is pair
        assert array_refs[0]() is not None
        worker.cancel()
        del output
        del pair

        def payload_collected():
            gc.collect()
            return array_refs[0]() is None and reference_pixels() is None and live_pixels() is None

        wait_for(payload_collected)
        assert worker.is_alive()
        assert not worker.busy
        assert worker.jobs_completed == 1
        assert len(array_refs) == 1  # Collection did not require another job.
    finally:
        stop(worker)


def test_frozen_pair_owns_readonly_pixels_independent_of_capture_buffer():
    capture_buffer = np.full((16, 16), 25, np.uint8)
    frame = FramePacket.from_array(capture_buffer)
    reference = ReferenceSnapshot(frame)
    capture_buffer[:] = 90
    assert np.all(reference.frame.pixels == 25)
    with pytest.raises(ValueError):
        reference.frame.pixels[0, 0] = 1
    with pytest.raises(FrozenInstanceError):
        reference.reference_id = "changed"
