"""Acquisition-thread/source contracts without camera hardware or a display."""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pytest

from measure_shift.models import FramePacket, MatchConfig, ReferenceSnapshot
from measure_shift.sources import DemoSource, ImageSource, SourceError
from measure_shift.workers import CaptureWorker, LatestValue, MeasurementWorker


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("Worker did not reach the expected integration state")
        threading.Event().wait(0.005)


def stop(*workers):
    for worker in workers:
        worker.stop()
    for worker in workers:
        worker.join(timeout=3)
        assert not worker.is_alive()


def test_real_demo_capture_measure_pause_resume_and_shutdown():
    frames = LatestValue()
    capture = CaptureWorker(DemoSource(fps=60), frames)
    measure = MeasurementWorker(frames)
    capture.start()
    measure.start()
    try:
        wait_for(lambda: frames.read()[0] is not None)
        initial, _ = frames.read()
        measure.configure(ReferenceSnapshot(initial), MatchConfig())
        wait_for(lambda: measure.output.read()[0] is not None)
        first, revision = measure.output.read()
        assert first.result.usable
        assert first.result.frame_id == first.frame.frame_id
        assert first.result.reference_id == first.reference.reference_id
        measure.paused.set()
        # Allow a measurement already executing at pause time to finish.
        threading.Event().wait(0.05)
        before_revision = measure.output.read()[1]
        capture_revision = frames.read()[1]
        wait_for(lambda: frames.read()[1] >= capture_revision + 4)
        assert measure.output.read()[1] == before_revision
        measure.paused.clear()
        wait_for(lambda: measure.output.read()[1] > before_revision)
        assert capture.error is None and measure.error is None
    finally:
        stop(capture, measure)
    assert capture.status == "Disconnected"


def test_image_sequence_finishes_and_releases_source(tmp_path):
    images = [np.full((16, 24), value, dtype=np.uint8) for value in [41, 72, 109]]
    for index, image in enumerate(images):
        assert cv2.imwrite(str(tmp_path / f"{index:02}.png"), image)
    source = ImageSource(tmp_path, fps=240, loop=False)
    frames = LatestValue()
    worker = CaptureWorker(source, frames)
    worker.start()
    worker.join(timeout=3)
    try:
        assert not worker.is_alive()
        assert worker.status == "End of input"
        assert worker.error is None
        final, revision = frames.read()
        assert revision == len(images)
        np.testing.assert_array_equal(final.pixels, images[-1])
        assert not final.pixels.flags.writeable
        assert not source._opened
    finally:
        stop(worker)


class GatedSource:
    name = "Gated camera"
    finite = False
    ended = False

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()
        self.calls = []
        self.value = 30.0
        self.fail_parameters = False
        self.fail_open = False
        self.buffer = np.full((12, 20), 57, np.uint8)

    def record(self, name):
        self.calls.append((name, threading.get_ident()))

    def open(self):
        self.record("open")
        if self.fail_open:
            raise SourceError("camera missing")

    def read(self, timeout_ms=200):
        self.record("read")
        self.entered.set()
        if not self.release.wait(timeout_ms / 1000):
            return None
        # Slow even after release so parameter/status assertions remain stable.
        threading.Event().wait(0.001)
        return self.buffer

    def controls(self):
        self.record("controls")
        return {"fps": {"min": 1.0, "max": 100.0, "value": self.value}}

    def set_parameters(self, **kwargs):
        self.record("parameters")
        if self.fail_parameters:
            raise SourceError("unsupported requested setting")
        self.value = float(kwargs.get("fps", self.value))

    def close(self):
        self.record("close")
        self.closed.set()


@pytest.mark.parametrize("failed_key", [None, "pixel_clock_mhz", "fps", "exposure_ms"])
def test_startup_maximizes_dependent_timing_before_first_frame(failed_key):
    class TimingCamera(GatedSource):
        def __init__(self):
            super().__init__()
            self.clock = 10
            self.exposure = 1
            self.requests = []
            self.first_read = None

        def controls(self):
            self.record("controls")
            return {
                "pixel_clock_mhz": {"values": [10, 40, 20], "value": self.clock},
                "fps": {"max": self.clock * 2, "value": self.value},
                "exposure_ms": {"max": 1000 / self.value, "value": self.exposure},
                "gain": {"max": 100, "value": 7},
            }

        def set_parameters(self, **kwargs):
            self.record("parameters")
            self.requests.append(kwargs)
            if failed_key in kwargs:
                raise SourceError("Device rejected setting")
            self.clock = kwargs.get("pixel_clock_mhz", self.clock)
            self.value = kwargs.get("fps", self.value)
            self.exposure = kwargs.get("exposure_ms", self.exposure)

        def read(self, timeout_ms=200):
            if self.first_read is None:
                self.first_read = (self.clock, self.value, self.exposure)
            return super().read(timeout_ms)

    source, frames = TimingCamera(), LatestValue()
    worker = CaptureWorker(source, frames, maximize_timing=True)
    source.release.set()
    worker.start()
    try:
        wait_for(lambda: frames.read()[0] is not None)
        clock = 10 if failed_key == "pixel_clock_mhz" else 40
        fps = 30 if failed_key == "fps" else clock * 2
        exposure = 1 if failed_key == "exposure_ms" else 1000 / fps
        assert source.requests == [
            {"pixel_clock_mhz": 40}, {"fps": clock * 2}, {"exposure_ms": 1000 / fps},
        ]
        assert source.first_read == (clock, fps, exposure)
        assert worker.controls["fps"]["value"] == fps
        assert worker.controls["exposure_ms"]["value"] == exposure
        assert worker.status == "Connected" and worker.error is None
        if failed_key:
            assert failed_key in worker.parameter_error
        else:
            assert worker.parameter_error is None
        assert all(thread == worker.ident for _, thread in source.calls)
    finally:
        stop(worker)


def test_stop_discards_frame_returned_after_stop_and_closes_on_capture_thread():
    source, frames = GatedSource(), LatestValue()
    worker = CaptureWorker(source, frames)
    worker.start()
    try:
        assert source.entered.wait(2)
        worker.stop()
        source.release.set()
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert frames.read() == (None, 0)
        assert source.closed.is_set()
        assert all(thread == worker.ident for _, thread in source.calls)
    finally:
        source.release.set()
        stop(worker)


def test_stop_during_frame_ownership_copy_cannot_publish_old_source_frame(monkeypatch):
    source, frames = GatedSource(), LatestValue()
    source.release.set()
    copy_started = threading.Event()
    finish_copy = threading.Event()
    original = FramePacket.from_array

    def gated_copy(cls, pixels, source_name="Image"):
        copy_started.set()
        assert finish_copy.wait(2)
        return original(pixels, source_name)

    monkeypatch.setattr(FramePacket, "from_array", classmethod(gated_copy))
    worker = CaptureWorker(source, frames)
    worker.start()
    try:
        assert copy_started.wait(2)
        worker.stop()
        frames.clear()  # Disconnect invalidates the old input before opening a new one.
        cleared_revision = frames.read()[1]
        finish_copy.set()
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert frames.read() == (None, cleared_revision)
        assert source.closed.is_set()
    finally:
        finish_copy.set()
        stop(worker)


def test_camera_parameters_are_queued_to_capture_thread_and_refresh_controls():
    source, frames = GatedSource(), LatestValue()
    worker = CaptureWorker(source, frames)
    worker.start()
    try:
        assert source.entered.wait(2)
        worker.set_parameters(fps=42)
        source.release.set()
        wait_for(lambda: worker.controls.get("fps", {}).get("value") == 42)
        assert worker.parameter_error is None
        assert worker.status == "Connected"
        assert all(thread == worker.ident for _, thread in source.calls)
    finally:
        source.release.set()
        stop(worker)


def test_parameter_error_does_not_disconnect_or_stop_acquisition():
    source, frames = GatedSource(), LatestValue()
    source.fail_parameters = True
    worker = CaptureWorker(source, frames)
    worker.start()
    try:
        assert source.entered.wait(2)
        worker.set_parameters(fps=42)
        source.release.set()
        wait_for(lambda: worker.parameter_error is not None)
        revision = frames.read()[1]
        wait_for(lambda: frames.read()[1] > revision)
        assert worker.status == "Connected"
        assert worker.error is None
    finally:
        source.release.set()
        stop(worker)


def test_open_error_closes_source_and_reports_disconnection():
    source, frames = GatedSource(), LatestValue()
    source.fail_open = True
    worker = CaptureWorker(source, frames)
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert source.closed.is_set()
    assert worker.status == "Disconnected"
    assert worker.error == "camera missing"
    assert frames.read() == (None, 0)


def test_capture_packets_own_data_even_when_source_reuses_buffer():
    source, frames = GatedSource(), LatestValue()
    worker = CaptureWorker(source, frames)
    worker.start()
    try:
        source.release.set()
        wait_for(lambda: frames.read()[0] is not None)
        first, _ = frames.read()
        source.buffer[:] = 192
        wait_for(lambda: np.all(frames.read()[0].pixels == 192))
        assert np.all(first.pixels == 57)
        assert not first.pixels.flags.writeable
    finally:
        source.release.set()
        stop(worker)


def test_partial_parameter_failure_still_refreshes_actual_camera_controls():
    class PartiallyAppliedSource(GatedSource):
        def set_parameters(self, **kwargs):
            self.value = float(kwargs["fps"])
            raise SourceError("exposure is unavailable")

    source, frames = PartiallyAppliedSource(), LatestValue()
    worker = CaptureWorker(source, frames)
    worker.start()
    try:
        assert source.entered.wait(2)
        worker.set_parameters(fps=42, exposure_ms=5)
        source.release.set()
        wait_for(lambda: worker.parameter_error is not None)
        assert worker.controls["fps"]["value"] == 42
        assert worker.error is None
    finally:
        source.release.set()
        stop(worker)


@pytest.fixture
def window():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from measure_shift.app import MainWindow

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    widget = MainWindow()
    yield application, widget
    widget.close()
    widget.deleteLater()
    application.processEvents()


def pump_until(application, widget, predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        application.processEvents()
        widget._poll()
        if time.monotonic() >= deadline:
            pytest.fail("UI did not reach the expected acquisition state")
        threading.Event().wait(0.005)


def test_reconnect_waits_for_previous_camera_release(window, monkeypatch):
    import measure_shift.app as app_module
    application, widget = window
    lock = threading.Lock()
    created = []

    class ExclusiveCamera(GatedSource):
        def __init__(self):
            super().__init__()
            self.owned = False

        def open(self):
            self.owned = lock.acquire(blocking=False)
            if not self.owned:
                raise SourceError("Camera already in use")
            super().open()

        def close(self):
            if self.owned:
                lock.release()
                self.owned = False
            super().close()

    def create(*args, **kwargs):
        source = ExclusiveCamera()
        created.append(source)
        return source

    monkeypatch.setattr(app_module, "create_source", create)
    widget.start_source("Basler")
    assert created[0].entered.wait(2)
    widget.start_source("Basler")
    pump_until(application, widget, lambda: len(created) == 2 and widget.capture_worker is not None and widget.capture_worker.status == "Connected")
    assert created[0].closed.is_set()
    assert widget.capture_worker.error is None


def test_capture_reference_while_paused_uses_visible_frame(window):
    _, widget = window
    first = FramePacket.from_array(np.full((48, 64), 41, np.uint8), "Visible frame")
    unseen = FramePacket.from_array(np.full((48, 64), 190, np.uint8), "New acquisition")
    widget.latest_frame = first
    widget._render()
    widget._pause(True)
    widget.latest_frame = unseen
    widget.capture_reference()
    assert widget.reference.frame.frame_id == first.frame_id
    np.testing.assert_array_equal(widget.reference.frame.pixels, widget.live_pane.array)


def test_camera_apply_refreshes_dependent_ui_bounds(window, monkeypatch):
    import measure_shift.app as app_module
    application, widget = window

    class DynamicCamera(GatedSource):
        def controls(self):
            result = super().controls()
            result["exposure_ms"] = {"min": .1, "max": 1000.0 / self.value, "value": 2.0}
            return result

    source = DynamicCamera()
    source.release.set()
    monkeypatch.setattr(app_module, "create_source", lambda *args, **kwargs: source)
    widget.start_source("Basler")
    pump_until(application, widget, lambda: widget.camera_controls["fps"].isEnabled())
    assert widget.camera_controls["exposure_ms"].maximum() == 10
    widget.camera_controls["fps"].setValue(50)
    widget._apply_camera()
    pump_until(application, widget, lambda: widget.camera_controls["exposure_ms"].maximum() == 20.0)
    assert source.value == 50


def test_ids_timing_controls_apply_and_refresh_camera_readback(window, monkeypatch, tmp_path):
    import measure_shift.app as app_module
    from measure_shift.storage import load_profile

    application, widget = window

    class TimingCamera(GatedSource):
        def __init__(self):
            super().__init__()
            self.clock = 20
            self.exposure = 2.0
            self.requests = []

        def controls(self):
            self.record("controls")
            return {
                "pixel_clock_mhz": {"min": 10, "max": 40, "value": self.clock, "values": [10, 20, 40]},
                "fps": {"min": 1, "max": self.clock * 2, "value": self.value},
                "exposure_ms": {"min": .1, "max": 1000 / self.value, "value": self.exposure, "step": .125},
            }

        def set_parameters(self, **kwargs):
            self.record("parameters")
            self.requests.append(kwargs)
            self.clock = kwargs.get("pixel_clock_mhz", self.clock)
            self.value = min(kwargs.get("fps", self.value), self.clock * 2)
            self.exposure = min(kwargs.get("exposure_ms", self.exposure), 1000 / self.value)

    source = TimingCamera()
    source.release.set()
    monkeypatch.setattr(app_module, "create_source", lambda *args, **kwargs: source)
    widget.start_source("IDS")
    clock = widget.camera_controls["pixel_clock_mhz"]
    rate = widget.camera_controls["fps"]
    exposure = widget.camera_controls["exposure_ms"]
    pump_until(application, widget, clock.isEnabled)
    assert [clock.itemData(index) for index in range(clock.count())] == [10, 20, 40]
    assert clock.currentData() == 40
    assert rate.maximum() == 80
    assert rate.value() == 80
    assert exposure.value() == 12.5
    assert source.requests == [{"pixel_clock_mhz": 40}, {"fps": 80}, {"exposure_ms": 12.5}]
    assert exposure.singleStep() == .125
    assert not widget.camera_controls["gain"].isEnabled()

    clock.setCurrentIndex(clock.findData(10))
    widget._apply_camera()
    pump_until(application, widget, lambda: rate.maximum() == 20)
    assert source.requests[-1] == {"pixel_clock_mhz": 10}
    assert rate.value() == 20
    assert exposure.maximum() == 50
    assert clock.currentData() == 10

    rate.setValue(10)
    exposure.setValue(5.125)
    widget._apply_camera()
    pump_until(application, widget, lambda: exposure.maximum() == 100)
    assert source.requests[-1] == {"fps": 10, "exposure_ms": 5.125}
    assert exposure.value() == 5.125
    assert all(thread == widget.capture_worker.ident for _, thread in source.calls)

    profile_path = tmp_path / "ids-settings.json"
    monkeypatch.setattr(widget, "_save_path", lambda *args: str(profile_path))
    widget.save_settings()
    assert load_profile(profile_path)[3] == {"pixel_clock_mhz": 10, "fps": 10, "exposure_ms": 5.125}

    widget.disconnect_source()
    assert all(not control.isEnabled() for control in widget.camera_controls.values())


def test_unlisted_current_clock_does_not_send_an_invalid_camera_value(window, monkeypatch):
    import measure_shift.app as app_module

    application, widget = window

    class UnknownClockCamera(GatedSource):
        def controls(self):
            return {**super().controls(), "pixel_clock_mhz": {
                "min": 10, "max": 40, "value": 15, "values": [10, 20, 40],
            }}

        def set_parameters(self, **kwargs):
            if "pixel_clock_mhz" in kwargs:
                # Simulate a device reporting an unlisted current clock even
                # after accepting the startup request for a supported value.
                assert kwargs == {"pixel_clock_mhz": 40}
            super().set_parameters(**kwargs)

    source = UnknownClockCamera()
    source.release.set()
    monkeypatch.setattr(app_module, "create_source", lambda *args, **kwargs: source)
    widget.start_source("IDS")
    clock = widget.camera_controls["pixel_clock_mhz"]
    pump_until(application, widget, clock.isEnabled)
    assert clock.currentIndex() == -1
    widget.camera_controls["fps"].setValue(42)
    widget._apply_camera()
    pump_until(application, widget, lambda: source.value == 42)
    assert widget.capture_worker.parameter_error is None


def test_rounded_camera_readback_is_not_resent_and_bounds_stay_valid(window, monkeypatch):
    import measure_shift.app as app_module

    application, widget = window
    minimum = .1346683433345833

    class PreciseCamera(GatedSource):
        def controls(self):
            return {
                "fps": {"min": 1.7463893, "max": 3.5202878, "value": 3.0017012},
                "exposure_ms": {"min": minimum, "max": 332.6786913862145, "value": 332.6786913862145},
            }

    source = PreciseCamera()
    source.release.set()
    monkeypatch.setattr(app_module, "create_source", lambda *args, **kwargs: source)
    widget.start_source("IDS")
    pump_until(application, widget, lambda: widget.camera_controls["fps"].isEnabled())
    queued = []
    monkeypatch.setattr(widget.capture_worker, "set_parameters", lambda **kwargs: queued.append(kwargs))
    widget._apply_camera()
    assert queued == []

    exposure = widget.camera_controls["exposure_ms"]
    exposure.setValue(exposure.minimum())
    widget._apply_camera()
    assert queued == [{"exposure_ms": minimum}]


def test_profile_staging_does_not_override_later_camera_readback(window, monkeypatch):
    import measure_shift.app as app_module
    application, widget = window
    source = GatedSource()
    source.release.set()
    monkeypatch.setattr(app_module, "create_source", lambda *args, **kwargs: source)
    widget._profile_camera = {"fps": 30.0}
    widget.start_source("Basler")
    pump_until(application, widget, lambda: widget.camera_controls["fps"].isEnabled())
    assert source.value == widget.camera_controls["fps"].value() == 100
    assert widget._profile_camera == {}
    widget.camera_controls["fps"].setValue(60)
    widget._apply_camera()
    pump_until(application, widget, lambda: source.value == 60 and widget.camera_controls["fps"].value() == 60)
    # Poll another cycle so a late controls refresh cannot restore the profile.
    widget._poll()
    assert widget.camera_controls["fps"].value() == 60


def test_eof_updates_alignment_state_and_disables_dead_camera_controls(window, monkeypatch):
    import measure_shift.app as app_module
    application, widget = window

    class EndingVideo(GatedSource):
        finite = True

        def __init__(self):
            super().__init__()
            self.generated = False
            self.buffer = np.random.default_rng(17).integers(20, 230, (48, 64), dtype=np.uint8)

        def read(self, timeout_ms=200):
            if not self.generated:
                self.generated = True
                return self.buffer
            self.entered.set()
            if self.release.wait(timeout_ms / 1000):
                self.ended = True
            return None

    source = EndingVideo()
    monkeypatch.setattr(app_module, "create_source", lambda *args, **kwargs: source)
    widget.start_source("Video", path="fake.avi", auto_reference=True)
    pump_until(application, widget, lambda: widget.last_pair is not None and widget.last_pair.result.usable)
    assert widget.state.text() == "ALIGNED"
    assert widget.camera_controls["fps"].isEnabled()
    source.release.set()
    pump_until(application, widget, lambda: widget.state.text() == "INPUT ENDED")
    assert source.closed.is_set()
    assert all(not control.isEnabled() for control in widget.camera_controls.values())
