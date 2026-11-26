"""Real-worker integration for optional inspection."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PySide6 import QtWidgets
from PySide6 import QtCore

from measure_shift import analytics
from measure_shift.app import MainWindow
from measure_shift.engine import MeasurementEngine
from measure_shift.models import FramePacket, InspectionPair
from measure_shift.workers import AnalysisOutput


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(qapp):
    window = MainWindow()
    window.timer.stop()
    window.show()
    qapp.processEvents()
    yield window
    window.close()
    window.deleteLater()
    qapp.processEvents()


def spin(window, qapp, predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        window._poll()
        if predicate():
            return
        time.sleep(0.005)
    assert predicate(), "Background inspection did not reach the expected state"


def install_pair(window, shift=(0, 0), reference=None):
    if reference is None:
        reference = cv2.GaussianBlur(np.random.default_rng(12).integers(15, 240, (128, 160), dtype=np.uint8), (3, 3), 0)
    if window.reference is None:
        window.set_reference(FramePacket.from_array(reference))
    frame = FramePacket.from_array(cv2.warpAffine(reference, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]), reference.shape[::-1]))
    engine = MeasurementEngine()
    engine.set_reference(window.reference)
    pair = InspectionPair(window.reference, frame, window.config, engine.measure(frame, window.config))
    window.latest_frame = frame
    window.last_pair = pair
    return pair


@pytest.mark.parametrize("width", [1500, 1000], ids=["regular", "narrow"])
def test_measurement_updates_preserve_sidebar_and_image_geometry(window, qapp, width):
    pair = install_pair(window)
    valid = replace(pair.result, dx=.5, dy=.25, validity="valid", reasons=())
    warnings = (
        "Weak match response",
        "Too little overlapping image area",
        "Estimated translation increases the shared-area image difference",
    )
    long_error = "Reference and current frame must have equal native dimensions. " * 20
    transitions = [
        *((replace(valid, validity="uncertain", reasons=(warning,)), ("X   +0.5 px", "Y   +0.2 px")) for warning in warnings),
        (replace(valid, validity="uncertain", reasons=warnings), ("X   +0.5 px", "Y   +0.2 px")),
        (replace(valid, dx=float("nan"), dy=float("nan"), validity="invalid", reasons=(long_error,)), ("X   — px", "Y   — px")),
        (valid, ("X   +0.5 px", "Y   +0.2 px")),
        (None, ("X   — px", "Y   — px")),
    ]
    watched = (window.tolerance, window.area, window.reference_pane, window.live_pane)

    def geometry_after_layout():
        # Layout requests can be posted by a preceding size-hint change.
        for _ in range(3):
            qapp.processEvents()
        rectangles = []
        for widget in watched:
            origin = widget.mapTo(window, QtCore.QPoint(0, 0))
            rectangles.append((origin.x(), origin.y(), widget.width(), widget.height()))
        return rectangles

    original_capture = window.capture_worker
    window.capture_worker = SimpleNamespace(error=None, status="Streaming", controls={})
    try:
        window.resize(width, window.height())
        window.last_pair = replace(pair, result=valid)
        window._update_readout()
        baseline = geometry_after_layout()
        for result, expected_readout in transitions:
            window.last_pair = replace(pair, result=result) if result is not None else None
            window._update_readout()
            geometry = geometry_after_layout()
            for actual, expected in zip(geometry, baseline):
                assert actual == pytest.approx(expected, abs=1), expected_readout
            assert (window.x_value.text(), window.y_value.text()) == expected_readout
    finally:
        window.capture_worker = original_capture


def test_readout_smooths_unique_frames_and_resumes_without_old_samples(window, monkeypatch):
    window.tolerance.setValue(.5)
    pair = install_pair(window)
    now = [time.monotonic()]
    monkeypatch.setattr("measure_shift.app.time", SimpleNamespace(monotonic=lambda: now[0]))

    def show_reading(elapsed, dx, dy):
        now[0] += elapsed
        frame = replace(pair.frame, frame_id=f"readout-{now[0]}", received_at=now[0])
        result = replace(pair.result, frame_id=frame.frame_id, dx=dx, dy=dy, validity="valid")
        window.last_pair = replace(pair, frame=frame, result=result)
        window._update_readout()

    with monkeypatch.context() as context:
        context.setattr(window, "capture_worker", SimpleNamespace(error=None, status="Streaming", controls={}))
        show_reading(0, .24, -.24)
        assert (window.x_value.text(), window.y_value.text()) == ("X   +0.2 px", "Y   -0.2 px")
        show_reading(.05, .44, -.44)
        expected = ("X   +0.3 px", "Y   -0.3 px")
        assert (window.x_value.text(), window.y_value.text()) == expected
        # Display smoothing must not change the tolerance decision or the stored result.
        assert window.state.text() == "AUSGERICHTET"
        assert (window.last_pair.result.dx, window.last_pair.result.dy) == (.44, -.44)
        for _ in range(8):
            window._update_readout()
        assert (window.x_value.text(), window.y_value.text()) == expected

        show_reading(.05, .64, -.64)
        assert window.state.text() == "AUSSER TOLERANZ"
        assert (window.x_value.text(), window.y_value.text()) == ("X   +0.4 px", "Y   -0.4 px")
        show_reading(.08, .64, -.64)
        expected = ("X   +0.6 px", "Y   -0.6 px")
        assert (window.x_value.text(), window.y_value.text()) == expected

        window._pause(True)
        now[0] += .4
        window._update_readout()
        assert (window.x_value.text(), window.y_value.text()) == expected
        window._pause(False)
        show_reading(.01, .84, -.84)
        assert (window.x_value.text(), window.y_value.text()) == ("X   +0.8 px", "Y   -0.8 px")
        show_reading(.2, -.01, .01)
        assert (window.x_value.text(), window.y_value.text()) == ("X   +0.0 px", "Y   +0.0 px")


@pytest.mark.parametrize("reset", ["reference", "config", "missing", "uncertain", "invalid"])
def test_readout_restarts_after_context_changes_or_unusable_measurements(window, monkeypatch, reset):
    pair = install_pair(window)
    now = [time.monotonic()]
    monkeypatch.setattr("measure_shift.app.time", SimpleNamespace(monotonic=lambda: now[0]))
    window.last_pair = replace(pair, result=replace(pair.result, dx=.4, dy=.8, validity="valid"))
    window._update_readout()

    if reset == "reference":
        window.set_reference(pair.reference.frame)
    elif reset == "config":
        window._set_config(window.config)
    else:
        window.last_pair = {
            "missing": None,
            "uncertain": replace(pair, result=replace(pair.result, dx=100, dy=100, validity="uncertain")),
            "invalid": replace(pair, result=replace(pair.result, dx=float("nan"), dy=float("nan"), validity="invalid")),
        }[reset]
        window._update_readout()
        expected = ("X   +100.0 px", "Y   +100.0 px") if reset == "uncertain" else ("X   — px", "Y   — px")
        assert (window.x_value.text(), window.y_value.text()) == expected

    now[0] += .02
    frame = replace(pair.frame, frame_id="after-reset")
    result = replace(pair.result, frame_id=frame.frame_id, reference_id=window.reference.reference_id,
                     config_id=window.config.config_id, dx=2, dy=-2, validity="valid")
    window.last_pair = replace(pair, reference=window.reference, config=window.config, frame=frame, result=result)
    window._update_readout()
    assert (window.x_value.text(), window.y_value.text()) == ("X   +2.0 px", "Y   -2.0 px")


def test_off_and_hidden_dock_do_no_optional_diagnostic_work(window, qapp):
    install_pair(window)
    called = []
    original = analytics.analyze_pair
    window.analytics._analyzer = lambda pair, tab: (called.append((pair.key, tab)), original(pair, tab))[1]
    for _ in range(3):
        window._schedule_analytics()
    assert not called and window.analytics._pending is None
    window.dock.show()
    window.dock.mode.setCurrentText("Live")
    spin(window, qapp, lambda: window.last_diagnostic is not None)
    assert len(called) == 1
    window.dock.hide()
    qapp.processEvents()
    window._schedule_analytics()
    assert window.last_diagnostic is None
    assert all(not page.views and page.snapshot is None for page in window.dock.pages.values())
    assert len(called) == 1 and window.analytics._pending is None


def test_collapsing_dock_drops_running_diagnostic_result(window, qapp):
    install_pair(window)
    entered, release = threading.Event(), threading.Event()
    original = analytics.analyze_pair

    def analyze(pair, tab):
        entered.set()
        release.wait(timeout=1)
        return original(pair, tab)

    window.analytics._analyzer = analyze
    window.dock.show()
    window.dock.mode.setCurrentText("Live")
    window._schedule_analytics()
    assert entered.wait(timeout=1)
    window.dock.hide()
    qapp.processEvents()
    release.set()
    spin(window, qapp, lambda: not window.analytics.busy)
    assert window.last_diagnostic is None
    assert window.analytics.output.read()[0] is None


def test_frozen_inspection_keeps_same_images_result_and_settings_across_tabs(window, qapp):
    captured = install_pair(window, (4, -3))
    window.capture_inspection()
    spin(window, qapp, lambda: window.last_diagnostic is not None)
    assert window.frozen_pair is captured
    assert window.last_diagnostic.pair is captured
    newer = install_pair(window, (9, 2))
    assert newer.key != captured.key
    window.dock.tabs.setCurrentIndex(2)  # Fourier
    spin(window, qapp, lambda: window.last_diagnostic is not None and window.last_diagnostic.tab == "Fourier")
    assert window.last_diagnostic.pair is captured
    assert window.last_diagnostic.pair.result.dx == captured.result.dx
    assert window.last_pair is newer
    assert window.analytics.jobs_completed == 2
    # Same reference/settings cannot make a different frame part of a frozen
    # inspection, even when an outdated result reaches the output mailbox.
    wrong = analytics.analyze_pair(newer, "Fourier")
    window.analytics.output.publish(AnalysisOutput(window.analytics.generation, wrong))
    window._poll()
    assert window.last_diagnostic.pair is captured
