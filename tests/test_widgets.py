"""Offscreen checks for native image coordinates and interactive analytics."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from measure_shift.analytics import analyze_pair
from measure_shift.engine import MeasurementEngine
from measure_shift.models import FramePacket, InspectionPair, MatchConfig, Region
from measure_shift.widgets import AnalysisPage, AnalyticsDock, ImagePane


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def show_widget(qapp):
    widgets = []

    def show(widget):
        widgets.append(widget)
        widget.resize(900, 700)
        widget.show()
        qapp.processEvents()
        return widget

    yield show
    for widget in widgets:
        widget.close()
        widget.deleteLater()
    qapp.processEvents()


def inspection():
    y, x = np.indices((64, 96))
    reference = (x + y * 2).astype(np.uint8)
    current = reference.copy()
    engine = MeasurementEngine()
    snapshot = engine.set_reference(FramePacket.from_array(reference))
    frame = FramePacket.from_array(current)
    config = MatchConfig()
    return InspectionPair(snapshot, frame, config, engine.measure(frame, config))


def test_image_cursor_and_roi_use_native_row_column_coordinates(show_widget, qapp):
    pane = show_widget(ImagePane("Image"))
    pair = inspection()
    pane.set_image(pair.frame.pixels)
    qapp.processEvents()
    point = pane.view.mapViewToScene(QtCore.QPointF(17.2, 11.2))
    pane._hover((point,))
    assert "x = 17" in pane.cursor.text()
    assert "y = 11" in pane.cursor.text()
    assert "intensity = 39" in pane.cursor.text()
    emitted = []
    pane.region_changed.connect(emitted.append)
    pane.set_region(Region("rectangle", 10, 12, 25, 20))
    pane.roi.setPos((15, 18))
    pane._roi_finished()
    assert emitted[-1] == Region("rectangle", 15, 18, 25, 20)


def test_drawn_line_profile_samples_both_native_images(show_widget):
    pair = inspection()
    page = show_widget(AnalysisPage("Image"))
    page.display(analyze_pair(pair, "Image"))
    distance, reference = page.profile_ref.getData()
    _, current = page.profile_live.getData()
    np.testing.assert_array_equal(reference, pair.reference.frame.pixels[32, 24:73])
    np.testing.assert_array_equal(current, pair.frame.pixels[32, 24:73])
    np.testing.assert_allclose(distance, np.arange(49))
    page.line_roi.setPos((0, 5))
    page._update_line()
    _, moved = page.profile_live.getData()
    np.testing.assert_array_equal(moved, pair.frame.pixels[37, 24:73])


def test_histogram_counts_and_correlation_curves_keep_signed_native_axes(show_widget, qapp):
    pair = inspection()
    image = show_widget(AnalysisPage("Image"))
    image.display(analyze_pair(pair, "Image"))
    xs, counts = image.views["histogram_reference"][1].getData()
    np.testing.assert_array_equal(xs, np.arange(256))
    assert counts.sum() == pair.frame.pixels.size
    page = show_widget(AnalysisPage("Correlation"))
    snapshot = analyze_pair(pair, "Correlation")
    page.display(snapshot)
    qapp.processEvents()
    xs, ys = page.views["correlation_horizontal"][1].getData()
    np.testing.assert_array_equal(xs, snapshot.metrics["slice_x_native_px"])
    np.testing.assert_array_equal(ys, snapshot.arrays["correlation_horizontal"])
    assert xs.min() < 0 < xs.max()
    assert "Measured displacement" in page.metrics.text()
    assert "Reconstructed" in page.metrics.text()
    for key, rect_key in (("reconstructed_correlation", "correlation_rect_native_px"), ("peak_zoom", "peak_zoom_rect_native_px")):
        pane = page.views[key]
        expected = snapshot.metrics[rect_key]
        rect = pane.image.mapRectToParent(pane.image.boundingRect())
        assert (rect.x(), rect.y(), rect.width(), rect.height()) == pytest.approx(expected)
        # The peak is centered on native shift zero, not an image array index.
        image = snapshot.arrays[key]
        y, x = np.unravel_index(np.argmax(image), image.shape)
        point = pane.image.mapToParent(QtCore.QPointF(x + 0.5, y + 0.5))
        assert (point.x(), point.y()) == pytest.approx((0, 0), abs=1e-6)
        for column in (x, x - 3):
            scene = pane.image.mapToScene(QtCore.QPointF(column + 0.5, y + 0.5))
            pane._hover((scene,))
            assert f"column {column}, row {y}" in pane.cursor.text()
            expected_intensity = f"{float(image[y, column]):.4g}"
            assert f"intensity = {expected_intensity}" in pane.cursor.text()
        assert "x = -3.000" in pane.cursor.text()


def test_analytics_dock_gates_visibility_enabled_tab_and_releases_views(show_widget, qapp):
    dock = show_widget(AnalyticsDock())
    assert not dock.active
    assert dock.rate.value() == 1
    dock.mode.setCurrentText("Live")
    assert dock.active
    page = dock.pages["Image"]
    dock.display(analyze_pair(inspection(), "Image"))
    assert page.snapshot is not None and page.views
    page.enabled.setChecked(False)
    assert not dock.active
    page.enabled.setChecked(True)
    dock.hide()
    qapp.processEvents()
    assert not dock.active
    dock.clear()
    assert page.snapshot is None and not page.views


def test_correlation_display_transform_preserves_signed_data_and_cursor(show_widget, qapp):
    snapshot = analyze_pair(inspection(), "Correlation")
    raw = snapshot.arrays["reconstructed_correlation"]
    before = {key: array.copy() for key, array in snapshot.arrays.items()}
    assert raw.min() < 0
    page = show_widget(AnalysisPage("Correlation"))
    page.display(snapshot)
    qapp.processEvents()
    pane = page.views["reconstructed_correlation"]
    assert pane.array is raw
    assert not np.shares_memory(pane.image.image, raw)
    assert pane.image.image.min() >= 0
    assert pane.image.image.max() == pytest.approx(1)
    row, column = np.unravel_index(np.argmin(raw), raw.shape)
    assert pane.image.image[row, column] == 0
    pane._hover((pane.image.mapToScene(QtCore.QPointF(column + 0.5, row + 0.5)),))
    assert f"intensity = {float(raw[row, column]):.4g}" in pane.cursor.text()
    assert "intensity = -" in pane.cursor.text()
    for key, array in snapshot.arrays.items():
        np.testing.assert_array_equal(array, before[key])
        assert not array.flags.writeable
    assert len(pane.annotations) == 2
    peak_x, peak_y = pane.annotations[0].getData()
    measured_x, measured_y = pane.annotations[1].getData()
    assert (peak_x[0], peak_y[0]) == pytest.approx((snapshot.metrics["reconstructed_integer_peak_dx_px"], snapshot.metrics["reconstructed_integer_peak_dy_px"]))
    assert (measured_x[0], measured_y[0]) == pytest.approx((snapshot.pair.result.dx, snapshot.pair.result.dy))
    page.display(snapshot)
    assert len(pane.annotations) == 2
