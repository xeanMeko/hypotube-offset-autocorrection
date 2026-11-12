"""Offscreen application workflows through real capture/measurement threads."""
from dataclasses import replace
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from measure_shift.app import MainWindow
from measure_shift.display import render_view
from measure_shift.engine import MeasurementEngine
from measure_shift.models import DiagnosticSnapshot, FramePacket, InspectionPair, MatchConfig, ReferenceSnapshot
from measure_shift.workers import AnalysisOutput


@pytest.fixture(scope="module")
def qt_application():
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    application.setQuitOnLastWindowClosed(False)
    yield application


def pump_until(application, predicate, timeout=6):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
        if predicate():
            return
        threading.Event().wait(0.005)
    pytest.fail("Application did not reach the expected state before its deadline")


def pump_for(application, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        application.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
        threading.Event().wait(0.005)


@pytest.fixture
def window(qt_application):
    widget = MainWindow()
    widget.show()
    qt_application.processEvents()
    yield widget
    widget.close()
    qt_application.processEvents()
    assert not widget.measurement.is_alive()
    assert not widget.analytics.is_alive()
    assert not widget.preview.is_alive()


def start_demo(window, application):
    window.start_source("Demo", auto_reference=True)
    pump_until(application, lambda: window.last_pair is not None)
    assert window.capture_worker.error is None
    assert window.last_pair.result.usable, window.last_pair.result.reasons


def wait_current_config(window, application):
    pump_until(application, lambda: window.last_pair is not None and
               window.last_pair.key[0] == window.reference.reference_id and
               window.last_pair.key[2] == window.config.config_id)


def test_demo_reference_roi_and_resolution_workflow(window, qt_application):
    start_demo(window, qt_application)
    assert window.live_pane.array.shape == (480, 640)
    assert window.analytics.jobs_completed == 0
    original_reference = window.reference.reference_id
    window.capture_reference()
    wait_current_config(window, qt_application)
    assert window.reference.reference_id != original_reference
    window.area.setCurrentText("Circle")
    wait_current_config(window, qt_application)
    assert window.config.region.kind == "circle"
    assert window.reference_pane._region == window.config.region
    window.area.setCurrentText("Rectangle")
    wait_current_config(window, qt_application)
    assert window.config.method == "phase"
    assert window.config.region.kind == "rectangle"
    window.scale.setCurrentIndex(2)
    wait_current_config(window, qt_application)
    assert window.last_pair.config.scale == 0.25
    assert window.last_pair.result.validity != "invalid", window.last_pair.result.reasons
    assert window.measurement.error is None


def test_live_frozen_and_off_analytics_keep_explicit_pair_provenance(window, qt_application):
    start_demo(window, qt_application)
    window.dock.show()
    window.dock.tabs.setCurrentIndex(5)  # Performance: no FFTs or image-grid allocation.
    window.dock.rate.setValue(10)
    window.dock.mode.setCurrentText("Live")
    pump_until(qt_application, lambda: window.last_diagnostic is not None)
    assert window.last_diagnostic.pair.key[0::2] == window.last_pair.key[0::2]
    window.capture_inspection()
    frozen = window.frozen_pair
    frozen_pixels = frozen.frame.pixels.copy()
    pump_until(qt_application, lambda: window.last_diagnostic is not None and
               window.last_diagnostic.pair.key == frozen.key)
    pump_until(qt_application, lambda: window.last_pair.frame.frame_id != frozen.frame.frame_id)
    assert window.frozen_pair is frozen
    np.testing.assert_array_equal(frozen.frame.pixels, frozen_pixels)
    window.dock.mode.setCurrentText("Off")
    jobs_completed = window.analytics.jobs_completed
    pump_for(qt_application, 0.12)
    assert window.last_diagnostic is None
    assert window.frozen_pair is None
    assert window.analytics.jobs_completed == jobs_completed
    assert all(page.snapshot is None for page in window.dock.pages.values())
    # A new configuration must produce fresh provenance when diagnostics resume.
    window.area.setCurrentText("Rectangle")
    wait_current_config(window, qt_application)
    window.dock.mode.setCurrentText("Live")
    pump_until(qt_application, lambda: window.last_diagnostic is not None and
               window.last_diagnostic.pair.config.config_id == window.config.config_id)


def test_pause_preserves_display_while_capture_continues_then_resumes(window, qt_application):
    start_demo(window, qt_application)
    window.pause_action.setChecked(True)
    paused_pair = window.last_pair
    paused_display = window.live_pane.array.copy()
    pump_until(qt_application, lambda: window.latest_frame.frame_id != paused_pair.frame.frame_id)
    pump_for(qt_application, 0.08)
    assert window.state.text() == "PAUSED"
    assert window.last_pair is paused_pair
    np.testing.assert_array_equal(window.live_pane.array, paused_display)
    displayed = window._displayed_frame
    assert displayed is not None
    assert displayed.frame_id != window.latest_frame.frame_id
    window.capture_reference()
    assert window.reference.frame.frame_id == displayed.frame_id
    window.pause_action.setChecked(False)
    pump_until(qt_application, lambda: window.last_pair is not None and
               window.last_pair.frame.frame_id != paused_pair.frame.frame_id)
    assert window.state.text() != "PAUSED"


def test_geometry_change_discards_reference_and_visible_old_measurement(window, qt_application):
    start_demo(window, qt_application)
    window.area.setCurrentText("Circle")
    wait_current_config(window, qt_application)
    window.capture_worker.stop()
    window.capture_worker.join(timeout=2)
    assert not window.capture_worker.is_alive()
    changed = FramePacket.from_array(np.full((96, 128), 120, np.uint8), "new geometry")
    window.frames.publish(changed)
    pump_until(qt_application, lambda: window.latest_frame is changed and window.reference is None)
    assert window.last_pair is None
    assert window.reference_pane.array is None
    assert window.config.region.kind == "full"
    assert "—" in window.x_value.text()
    assert "—" in window.y_value.text()
    assert "geometry changed" in window.notice.text().lower()


def test_live_compensated_preview_is_eventually_rendered_and_matches_its_pair(window, qt_application):
    start_demo(window, qt_application)
    window.view_mode.setCurrentText("Compensated preview")
    pump_until(qt_application, lambda: window.live_pane.array.ndim == 3)
    # Stop measurement changes while allowing the GUI/preview worker to run,
    # then verify that the displayed native array belongs to that exact pair.
    window.measurement.paused.set()
    pump_for(qt_application, 0.08)
    pair = window.last_pair
    pump_until(qt_application, lambda: pair.frame.frame_id[:8] in window.live_pane.title.text())
    np.testing.assert_array_equal(window.live_pane.array, render_view(pair, "Compensated preview"))


def test_frozen_inspector_rejects_wrong_frame_even_with_same_reference_and_config(window, qt_application):
    # Inject an unexpected completion after stopping real computation to check
    # the GUI acceptance boundary independently of queue cancellation.
    window.timer.stop()
    for worker in (window.measurement, window.analytics, window.preview):
        worker.stop()
        worker.join(timeout=2)
    rng = np.random.default_rng(24)
    source = FramePacket.from_array(rng.integers(0, 255, (64, 96), dtype=np.uint8))
    window.set_reference(source)
    window.config = MatchConfig()
    engine = MeasurementEngine()
    engine.set_reference(window.reference)
    frame = FramePacket.from_array(source.pixels)
    pair = InspectionPair(window.reference, frame, window.config, engine.measure(frame, window.config))
    foreign_frame = FramePacket.from_array(source.pixels)
    foreign = InspectionPair(window.reference, foreign_frame, window.config,
                             engine.measure(foreign_frame, window.config))
    window.latest_frame = frame
    window.last_pair = pair
    window.dock.tabs.setCurrentIndex(5)
    window.dock.show()
    window.dock.mode.setCurrentText("Frozen")
    assert window.frozen_pair is pair
    wrong_snapshot = DiagnosticSnapshot(foreign, "Performance")
    window.analytics.output.publish(AnalysisOutput(window.analytics.generation, snapshot=wrong_snapshot))
    window._poll()
    assert window.last_diagnostic is None
    correct_snapshot = DiagnosticSnapshot(pair, "Performance")
    window.analytics.output.publish(AnalysisOutput(window.analytics.generation, snapshot=correct_snapshot))
    window._poll()
    assert window.last_diagnostic is correct_snapshot
    window.dock.mode.setCurrentText("Live")
    assert window.last_diagnostic is None
    stale_config = replace(window.config, config_id="old configuration")
    old_reference = ReferenceSnapshot(source)
    stale_pairs = [
        replace(pair, config=stale_config, result=replace(pair.result, config_id=stale_config.config_id)),
        replace(pair, reference=old_reference, result=replace(pair.result, reference_id=old_reference.reference_id)),
    ]
    for stale_pair in stale_pairs:
        snapshot = DiagnosticSnapshot(stale_pair, "Performance")
        window.analytics.output.publish(AnalysisOutput(window.analytics.generation, snapshot=snapshot))
        window._poll()
        assert window.last_diagnostic is None
    # Live diagnostics can be an older frame, provided the displayed provenance
    # is explicit and its reference/settings still match.
    compatible_snapshot = DiagnosticSnapshot(foreign, "Performance")
    window.analytics.output.publish(AnalysisOutput(window.analytics.generation, snapshot=compatible_snapshot))
    window._poll()
    assert window.last_diagnostic is compatible_snapshot


def test_close_waits_for_export_and_delivers_completion_callback(window, qt_application):
    entered = threading.Event()
    release = threading.Event()
    callbacks = []

    def export():
        entered.set()
        assert release.wait(2)
        return "saved"

    window._task(export, callbacks.append, finish_before_close=True)
    try:
        assert entered.wait(2)
        assert not window.close()
        assert window.isVisible()
        assert window._close_pending
        assert not window._closing
        assert window.timer.isActive()
        assert callbacks == []
        release.set()
        pump_until(qt_application, lambda: not window.isVisible())
        assert callbacks == ["saved"]
        assert window._closing
    finally:
        release.set()


def test_export_failure_cancels_pending_close_and_remains_visible(window, qt_application):
    entered = threading.Event()
    release = threading.Event()

    def export():
        entered.set()
        assert release.wait(2)
        raise OSError("Export destination is not writable")

    window._task(export, lambda _: pytest.fail("Failed export invoked success callback"), finish_before_close=True)
    try:
        assert entered.wait(2)
        assert not window.close()
        release.set()
        pump_until(qt_application, lambda: not window._close_pending)
        assert window.isVisible()
        assert not window._closing
        assert "not writable" in window.notice.text()
    finally:
        release.set()
