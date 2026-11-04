"""Real-worker integration for optional inspection and the beta slot workflow."""
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
from measure_shift.slots import SlotCandidate
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
def test_match_warnings_preserve_sidebar_and_image_geometry(window, qapp, width):
    pair = install_pair(window)
    valid = replace(pair.result, dx=.5, dy=.25, validity="valid", reasons=())
    warnings = (
        "Weak match response",
        "Too little overlapping image area",
        "Estimated translation increases the shared-area image difference",
    )
    long_error = "Reference and current frame must have equal native dimensions. " * 20
    transitions = [
        *((replace(valid, validity="uncertain", reasons=(warning,)), warning) for warning in warnings),
        (replace(valid, validity="uncertain", reasons=warnings), "\n".join(warnings)),
        (replace(valid, dx=float("nan"), dy=float("nan"), validity="invalid", reasons=(long_error,)), long_error),
        (valid, "+X right · +Y down\nDisplacement is reference → current."),
        (None, "+X right · +Y down\nDisplacement is reference → current."),
    ]
    watched = (window.tolerance, window.find_slot, window.reference_pane, window.live_pane)

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
        assert window.reasons.isReadOnly()
        for result, expected_message in transitions:
            window.last_pair = replace(pair, result=result) if result is not None else None
            window._update_readout()
            geometry = geometry_after_layout()
            for actual, expected in zip(geometry, baseline):
                assert actual == pytest.approx(expected, abs=1), expected_message
            assert window.reasons.toPlainText() == expected_message
            if expected_message == long_error:
                scrollbar = window.reasons.verticalScrollBar()
                assert scrollbar.maximum() > 0
                scrollbar.setValue(scrollbar.maximum())
                assert scrollbar.value() > 0
    finally:
        window.capture_worker = original_capture


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


def test_beta_find_choose_track_and_loss_use_fixed_reference(window, qapp):
    image = np.full((320, 320), 8, np.uint8)
    cv2.circle(image, (160, 160), 142, 175, -1)
    cv2.rectangle(image, (150, 70), (166, 200), 20, -1)
    window.latest_frame = FramePacket.from_array(image)
    window.capture_reference()
    original_reference_id = window.reference.reference_id
    window._find_slots()
    spin(window, qapp, lambda: bool(window._slot_candidates))
    assert window.slot_choice.count() == len(window._slot_candidates) == 1
    window._use_slot()
    assert window.slot_tracking.isChecked()
    moved = cv2.warpAffine(image, np.float32([[1, 0, 9], [0, 1, -6]]), image.shape[::-1])
    window.frames.publish(FramePacket.from_array(moved))
    spin(window, qapp, lambda: window.last_pair is not None)
    result = window.last_pair.result
    assert result.usable, result.reasons
    assert (result.dx, result.dy) == pytest.approx((9, -6), abs=0.3)
    assert window.last_pair.config.live_region is not None
    assert window.reference.reference_id == original_reference_id
    assert window.measurement.slot_status == "Tracking"
    window.frames.publish(FramePacket.from_array(np.zeros_like(image)))
    spin(window, qapp, lambda: window.last_pair is not None and not window.last_pair.result.usable)
    assert window.measurement.slot_status == "Lost"
    assert "lost" in window.last_pair.result.reasons[0]
    assert window.reference.reference_id == original_reference_id


def test_beta_tracking_without_selection_returns_to_off(window):
    window.slot_tracking.setChecked(True)
    assert not window.slot_tracking.isChecked()
    assert window.measurement._slot_selection is None
    assert "select a candidate" in window.notice.text()


def test_beta_discards_candidates_from_an_outdated_analysis_region(window, qapp, monkeypatch):
    install_pair(window)
    entered, release = threading.Event(), threading.Event()

    def detect(image, region):
        entered.set()
        release.wait(timeout=1)
        return [SlotCandidate(1, (20, 20, 40, 60), (40, 50), 1)]

    monkeypatch.setattr("measure_shift.slots.detect_slots", detect)
    window._find_slots()
    assert entered.wait(timeout=1)
    window.area.setCurrentText("Rectangle")
    release.set()
    spin(window, qapp, lambda: not window.tasks)
    assert window.find_slot.isEnabled()
    assert not window._slot_candidates
    assert window.slot_choice.count() == 0
