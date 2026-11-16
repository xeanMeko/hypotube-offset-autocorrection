"""Interactive image views and the optional analytics dock."""
from __future__ import annotations

import math
import time

import cv2
import numpy as np
# QtWidgets must precede QtGui/pyqtgraph to avoid mixed Conda/system X11 libs.
from PySide6 import QtWidgets
from PySide6 import QtCore
import pyqtgraph as pg

from .models import DiagnosticSnapshot, Region

pg.setConfigOptions(imageAxisOrder="row-major", antialias=False, background="#101820", foreground="#d7e2eb")


class MessageStatusBar(QtWidgets.QStatusBar):
    """Show temporary messages in a shared area without hiding other readouts."""

    def __init__(self, fallback, parent=None):
        super().__init__(parent)
        self.setSizeGripEnabled(False)
        self._fallback = fallback
        self._message = QtWidgets.QLabel()
        self._message.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self._message.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self._message.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self.message_area = QtWidgets.QStackedWidget()
        self.message_area.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self.message_area.addWidget(fallback)
        self.message_area.addWidget(self._message)
        self._message_timer = QtCore.QTimer(self)
        self._message_timer.setSingleShot(True)
        self._message_timer.timeout.connect(self.clearMessage)

    def showMessage(self, message, timeout=0):
        previous = self.currentMessage()
        self._message_timer.stop()
        self._message.setText(message)
        self._message.setToolTip(message)
        self.message_area.setCurrentWidget(self._message if message else self._fallback)
        if message and timeout > 0:
            self._message_timer.start(timeout)
        if message != previous:
            self.messageChanged.emit(message)

    def clearMessage(self):
        self.showMessage("")

    def currentMessage(self):
        return self._message.text()


class TimedPlotWidget(pg.PlotWidget):
    def __init__(self, *args, **kwargs):
        self.last_paint_ms = 0.0
        super().__init__(*args, **kwargs)

    def paintEvent(self, event):
        started = time.perf_counter()
        super().paintEvent(event)
        self.last_paint_ms = (time.perf_counter() - started) * 1000


class ImagePane(QtWidgets.QWidget):
    region_changed = QtCore.Signal(object)
    crosshair_moved = QtCore.Signal(object)

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.title = QtWidgets.QLabel(title)
        self.title.setObjectName("paneTitle")
        layout.addWidget(self.title)
        self.plot = TimedPlotWidget()
        self.plot.setMinimumSize(180, 160)
        self.plot.setLabel("bottom", "x", units="px")
        self.plot.setLabel("left", "y", units="px")
        self.view = self.plot.getViewBox()
        self.view.setAspectLocked(True)
        self.view.invertY(True)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image)
        layout.addWidget(self.plot)
        self.cursor = QtWidgets.QLabel("Scroll to zoom · drag to pan · right-click for view options")
        self.cursor.setObjectName("muted")
        layout.addWidget(self.cursor)
        self.array = None
        self._display_scale = None
        self.view.sigResized.connect(self._apply_display_scale)
        self.view.sigRangeChangedManually.connect(self._remember_display_scale)
        self.roi = None
        self._region = None
        self._auto_done = False
        self._crosshair_visible = False
        self._crosshair_movable = False
        self._crosshair_position = None
        self.crosshair_horizontal = pg.PlotCurveItem(pen=pg.mkPen("#66d9ef", width=1, cosmetic=True))
        self.crosshair_vertical = pg.PlotCurveItem(pen=pg.mkPen("#66d9ef", width=1, cosmetic=True))
        for line in (self.crosshair_horizontal, self.crosshair_vertical):
            line.setZValue(1)
            line.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
            line.setAcceptHoverEvents(False)
            self.plot.addItem(line, ignoreBounds=True)
            line.hide()
        self._mouse = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=20, slot=self._hover)
        self.plot.scene().sigMouseClicked.connect(self._place_crosshair)
        self.arrow_line = pg.PlotDataItem(pen=pg.mkPen("#ffcd66", width=2))
        self.arrow = pg.ArrowItem(angle=0, headLen=14, tailLen=0, brush="#ffcd66", pen="#ffcd66")
        self.plot.addItem(self.arrow_line)
        self.plot.addItem(self.arrow)
        self.arrow.hide()
        self.arrow_line.hide()
        self.annotations = []

    def set_image(self, array: np.ndarray, *, auto=False, color=False, power_scale=None, power_max=None) -> None:
        previous_shape = None if self.array is None else self.array.shape
        self.array = array
        if array.dtype == np.uint8:
            levels = (0, 255)
        else:
            finite = array[np.isfinite(array)]
            if finite.size:
                levels = tuple(float(x) for x in np.percentile(finite[::max(1, finite.size // 200000)], (0.5, 99.5)))
                if levels[1] <= levels[0]:
                    levels = (levels[0], levels[0] + 1)
            else:
                levels = (0, 1)
        if array.ndim == 3 and array.dtype != np.uint8:
            levels = (0, 1)
        display = array
        if power_scale is not None:
            maximum = max(float(power_max if power_max is not None else np.max(array)), 1e-12)
            display = (np.maximum(array, 0) / maximum) ** power_scale
            levels = (0, 1)
        self.image.setLookupTable(pg.colormap.get("viridis").getLookupTable(nPts=256) if color and array.ndim == 2 else None)
        self.image.setImage(display, autoLevels=False, levels=levels)
        if previous_shape != array.shape:
            self._update_crosshair()
        if auto or not self._auto_done or previous_shape != array.shape:
            h, w = array.shape[:2]
            if self._display_scale is None:
                self.view.setRange(rect=QtCore.QRectF(0, 0, w, h), padding=.03)
            else:
                self._apply_display_scale(center=QtCore.QPointF(w / 2, h / 2))
            self._auto_done = True

    def clear_image(self) -> None:
        self.array = None
        self.image.clear()
        self._update_crosshair()

    @property
    def display_scale(self) -> float | None:
        return self._display_scale

    def set_display_scale(self, scale: float | None) -> None:
        """Set screen pixels per image pixel; None fits the whole image."""
        self._display_scale = scale
        if self.array is None:
            return
        if scale is None:
            h, w = self.array.shape[:2]
            self.view.setRange(rect=QtCore.QRectF(0, 0, w, h), padding=.03)
        else:
            self._apply_display_scale()

    def _apply_display_scale(self, *_, center=None) -> None:
        if self._display_scale is None or self.array is None:
            return
        bounds = self.view.rect()
        if bounds.width() <= 0 or bounds.height() <= 0:
            return
        center = self.view.viewRect().center() if center is None else center
        width = bounds.width() / self._display_scale
        height = bounds.height() / self._display_scale
        self.view.setRange(rect=QtCore.QRectF(center.x() - width / 2, center.y() - height / 2, width, height), padding=0)

    def _remember_display_scale(self, *_) -> None:
        if self.array is not None:
            self._display_scale = self.view.rect().width() / self.view.viewRect().width()

    def set_crosshair_visible(self, visible: bool) -> None:
        self._crosshair_visible = visible
        if not visible:
            self.set_crosshair_movable(False)
        self._update_crosshair()

    @property
    def crosshair_position(self) -> tuple[float, float] | None:
        """Native image coordinates, or None to follow the image center."""
        return self._crosshair_position

    def set_crosshair_position(self, position: tuple[float, float] | None) -> None:
        self._crosshair_position = position
        self._update_crosshair()

    def set_crosshair_movable(self, movable: bool) -> None:
        self._crosshair_movable = movable and self._crosshair_visible
        if self._crosshair_movable:
            self.plot.viewport().setCursor(QtCore.Qt.CursorShape.CrossCursor)
        else:
            self.plot.viewport().unsetCursor()

    def _place_crosshair(self, event) -> None:
        if (not self._crosshair_movable or self.array is None
                or event.button() != QtCore.Qt.MouseButton.LeftButton
                or not self.view.sceneBoundingRect().contains(event.scenePos())):
            return
        point = self.image.mapFromScene(event.scenePos())
        x, y = point.x(), point.y()
        h, w = self.array.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            self.set_crosshair_position((x, y))
            self.crosshair_moved.emit(self._crosshair_position)

    def _update_crosshair(self) -> None:
        visible = self._crosshair_visible and self.array is not None
        if visible:
            h, w = self.array.shape[:2]
            x, y = self._crosshair_position or (w / 2, h / 2)
            # Keep a saved native position when switching image sizes, while
            # keeping the rendered guides inside a smaller image.
            x, y = min(x, w - .5), min(y, h - .5)
            self.crosshair_horizontal.setData([0, w], [y, y])
            self.crosshair_vertical.setData([x, x], [0, h])
        self.crosshair_horizontal.setVisible(visible)
        self.crosshair_vertical.setVisible(visible)

    def _hover(self, args) -> None:
        if self.array is None:
            return
        item_point = self.image.mapFromScene(args[0])
        point = self.image.mapToParent(item_point)
        x, y = int(math.floor(item_point.x())), int(math.floor(item_point.y()))
        if 0 <= y < self.array.shape[0] and 0 <= x < self.array.shape[1]:
            value = self.array[y, x]
            text = np.array2string(value, precision=3) if np.ndim(value) else f"{float(value):.4g}"
            coordinates = f"x = {x}   y = {y}" if self.image.transform().isIdentity() else f"x = {point.x():+.3f}   y = {point.y():+.3f}   [column {x}, row {y}]"
            self.cursor.setText(f"{coordinates}   intensity = {text}")

    def set_region(self, region: Region) -> None:
        if self.array is None and region.kind != "full":
            self._region = None
            return
        if self._region == region:
            return
        if self.roi is not None:
            self.plot.removeItem(self.roi)
            self.roi.deleteLater()
            self.roi = None
        self._region = region
        if region.kind == "full" or self.array is None:
            return
        h, w = self.array.shape[:2]
        options = dict(pos=(region.x, region.y), size=(region.width, region.height),
                       pen=pg.mkPen("#ffcd66", width=2), maxBounds=QtCore.QRectF(0, 0, w, h),
                       translateSnap=True, scaleSnap=True)
        if region.kind == "circle":
            self.roi = pg.CircleROI(**options)
        else:
            self.roi = pg.RectROI(**options, sideScalers=True)
        self.plot.addItem(self.roi)
        self.roi.sigRegionChangeFinished.connect(self._roi_finished)

    def _roi_finished(self) -> None:
        if self.roi is None or self.array is None:
            return
        pos, size = self.roi.pos(), self.roi.size()
        h, w = self.array.shape[:2]
        x, y = max(0, int(round(pos.x()))), max(0, int(round(pos.y())))
        rw, rh = max(8, int(round(size.x()))), max(8, int(round(size.y())))
        rw, rh = min(rw, w), min(rh, h)
        x, y = min(x, w - rw), min(y, h - rh)
        if self._region.kind == "circle":
            rw = rh = min(rw, rh)
        self.region_changed.emit(Region(self._region.kind, x, y, rw, rh))

    def show_correction(self, dx: float | None, dy: float | None) -> None:
        if self.array is None or dx is None or dy is None or not np.isfinite([dx, dy]).all():
            self.arrow.hide()
            self.arrow_line.hide()
            return
        h, w = self.array.shape[:2]
        cx, cy = w / 2, h / 2
        correction = np.array([-dx, -dy], dtype=float)
        length = np.linalg.norm(correction)
        if length < 0.05:
            self.arrow.hide()
            self.arrow_line.hide()
            return
        correction *= min(1, min(w, h) * 0.3 / length)
        ex, ey = cx + correction[0], cy + correction[1]
        self.arrow_line.setData([cx, ex], [cy, ey])
        self.arrow.setPos(ex, ey)
        self.arrow.setStyle(angle=180 + math.degrees(math.atan2(correction[1], correction[0])))
        self.arrow.show()
        self.arrow_line.show()

    def clear_annotations(self) -> None:
        for item in self.annotations:
            self.plot.removeItem(item)
        self.annotations.clear()


class AnalysisPage(QtWidgets.QWidget):
    def __init__(self, tab: str):
        super().__init__()
        self.tab = tab
        outer = QtWidgets.QVBoxLayout(self)
        self.enabled = QtWidgets.QCheckBox("Enable this analysis")
        self.enabled.setChecked(True)
        outer.addWidget(self.enabled)
        self.metrics = QtWidgets.QLabel("Enable Live analytics or capture a frozen pair to inspect this view.")
        self.metrics.setWordWrap(True)
        self.metrics.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(self.metrics)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.contents = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.contents)
        self.scroll.setWidget(self.contents)
        outer.addWidget(self.scroll, 1)
        self.views: dict = {}
        self.snapshot = None
        self.line_roi = None
        self.profile = None
        self._columns = 0
        self.scroll.viewport().installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is self.scroll.viewport() and event.type() == QtCore.QEvent.Type.Resize:
            self._layout_views()
        return super().eventFilter(watched, event)

    def _layout_views(self, *, force=False) -> None:
        if not self.views:
            return
        margins = self.grid.contentsMargins()
        width = self.scroll.viewport().width() - margins.left() - margins.right()
        spacing = self.grid.horizontalSpacing()
        columns = max(1, min(len(self.views), (width + spacing) // (380 + spacing)))
        if columns == self._columns and not force:
            return
        # Move existing views so resizing preserves zoom, curves, and the line ROI.
        while self.grid.count():
            self.grid.takeAt(0)
        for column in range(max(columns, self._columns)):
            self.grid.setColumnStretch(column, 1 if column < columns else 0)
        self._columns = columns
        for i, view in enumerate(self.views.values()):
            widget = view[0] if isinstance(view, tuple) else view
            self.grid.addWidget(widget, i // columns, i % columns)
        if self.profile is not None:
            self.grid.addWidget(self.profile, (len(self.views) + columns - 1) // columns, 0, 1, columns)

    def clear(self) -> None:
        self.snapshot = None
        self.line_roi = None
        self.profile = None
        for view in self.views.values():
            if isinstance(view, ImagePane):
                view.clear_image()
            elif isinstance(view, tuple):
                view[1].setData([], [])
        self.views = {}
        while self.grid.count():
            child = self.grid.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.metrics.setText("Analytics paused. No diagnostic arrays retained in this view.")

    def display(self, snapshot: DiagnosticSnapshot) -> None:
        self.snapshot = snapshot
        pair, result = snapshot.pair, snapshot.pair.result
        values = []
        for key, value in snapshot.metrics.items():
            if isinstance(value, (tuple, list, np.ndarray)) and np.size(value) > 8:
                continue
            if isinstance(value, float):
                value = f"{value:.5g}"
            values.append(f"{key.replace('_', ' ')}: {value}")
        provenance = f"Frame {pair.frame.frame_id[:8]} · reference {pair.reference.reference_id[:8]} · captured {time.strftime('%H:%M:%S', time.localtime(pair.frame.captured_at))} · analysis scale {pair.config.scale:g}"
        actual = f"Measured displacement: ({result.dx:+.3f}, {result.dy:+.3f}) px · response {result.response:.4f} · {result.validity}"
        self.metrics.setText(provenance + "\n" + actual + "\n" + "  |  ".join(values))
        arrays = {key: value for key, value in snapshot.arrays.items() if isinstance(value, np.ndarray) and value.ndim in (1, 2, 3)}
        if list(self.views) != list(arrays):
            # Keep only the active module's arrays and widgets.
            self.clear()
            self.snapshot = snapshot
            self.metrics.setText(provenance + "\n" + actual + "\n" + "  |  ".join(values))
            for key, array in arrays.items():
                title = key.replace('_', ' ').capitalize()
                if array.ndim == 1:
                    widget = pg.PlotWidget(title=title)
                    widget.setMinimumHeight(200)
                    widget.showGrid(x=True, y=True, alpha=0.2)
                    widget.setLabel("left", "Value")
                    widget.setLabel("bottom", "Sample")
                    curve = widget.plot(pen=pg.mkPen("#69dfb7", width=2))
                    self.views[key] = (widget, curve)
                else:
                    widget = ImagePane(title)
                    widget.setMinimumHeight(280)
                    widget.cursor.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
                    widget.cursor.setToolTip(widget.cursor.text())
                    self.views[key] = widget
            self._layout_views(force=True)
        for key, array in arrays.items():
            view = self.views[key]
            if isinstance(view, tuple):
                widget, curve = view
                axis_key = "slice_x_native_px" if "horizontal" in key else "slice_y_native_px" if "vertical" in key else ""
                axis = snapshot.metrics.get(axis_key)
                if axis is not None and len(axis) == array.size:
                    curve.setData(np.asarray(axis), array)
                    widget.setLabel("bottom", "Displacement", units="px")
                else:
                    curve.setData(array)
                    widget.setLabel("bottom", "Intensity" if "histogram" in key else "Sample")
            else:
                color = any(token in key for token in ("magnitude", "phase", "correlation", "difference", "centered"))
                is_map = key in ("reconstructed_correlation", "peak_zoom")
                view.set_image(array, color=color or is_map,
                               power_scale=.35 if is_map else None,
                               power_max=snapshot.metrics.get("reconstructed_peak_value") if is_map else None)
                rect_key = "correlation_rect_native_px" if key == "reconstructed_correlation" else "peak_zoom_rect_native_px" if key == "peak_zoom" else None
                if rect_key is not None and rect_key in snapshot.metrics:
                    view.image.setRect(QtCore.QRectF(*snapshot.metrics[rect_key]))
                    if not getattr(view, "_signed_rect_initialized", False):
                        view.view.autoRange(padding=.03)
                        view._signed_rect_initialized = True
                if "phase" in key or "magnitude" in key or "spectrum" in key:
                    view.plot.setLabel("bottom", "Frequency bin u")
                    view.plot.setLabel("left", "Frequency bin v")
                if key in ("reconstructed_correlation", "peak_zoom"):
                    view.plot.setLabel("bottom", "Displacement x", units="px")
                    view.plot.setLabel("left", "Displacement y", units="px")
                    view.clear_annotations()
                    integer_peak = pg.ScatterPlotItem([snapshot.metrics["reconstructed_integer_peak_dx_px"]],
                        [snapshot.metrics["reconstructed_integer_peak_dy_px"]], symbol="o", size=13,
                        pen=pg.mkPen("#ffffff", width=2), brush=None)
                    measured = pg.ScatterPlotItem([result.dx], [result.dy], symbol="+", size=15,
                        pen=pg.mkPen("#62d4ef", width=2))
                    for marker in (integer_peak, measured):
                        view.plot.addItem(marker)
                        view.annotations.append(marker)
                    view.cursor.setText("Positive correlation shown with power scale 0.35 · cursor and slices retain raw signed values · cyan + measured shift")
                    view.cursor.setToolTip(view.cursor.text())
        if self.tab == "Image":
            self._add_line_inspector()

    def _add_line_inspector(self) -> None:
        if self.snapshot is None:
            return
        panes = [view for view in self.views.values() if isinstance(view, ImagePane)]
        if not panes:
            return
        if self.line_roi is None:
            h, w = self.snapshot.pair.frame.shape
            self.line_roi = pg.LineSegmentROI([(w * .25, h * .5), (w * .75, h * .5)], pen=pg.mkPen("#ffcd66", width=2))
            panes[-1].plot.addItem(self.line_roi)
            self.profile = pg.PlotWidget(title="Draw the yellow line to inspect intensity across a feature")
            self.profile.setMinimumHeight(220)
            self.profile.addLegend()
            self.profile.setLabel("bottom", "Distance along line", units="px")
            self.profile.setLabel("left", "Intensity (0–255)")
            self.profile_ref = self.profile.plot(name="Reference", pen=pg.mkPen("#ff7979", width=2))
            self.profile_live = self.profile.plot(name="Current", pen=pg.mkPen("#62d4ef", width=2))
            self._layout_views(force=True)
            self.line_roi.sigRegionChanged.connect(self._update_line)
        self._update_line()

    def _update_line(self) -> None:
        if self.line_roi is None or self.snapshot is None:
            return
        handles = self.line_roi.getLocalHandlePositions()
        points = [self.line_roi.mapToParent(pos) for _, pos in handles]
        if len(points) != 2:
            return
        a, b = points
        length = math.hypot(b.x() - a.x(), b.y() - a.y())
        count = max(2, min(4096, int(length) + 1))
        xs = np.linspace(a.x(), b.x(), count).astype(np.float32)[None, :]
        ys = np.linspace(a.y(), b.y(), count).astype(np.float32)[None, :]
        distance = np.linspace(0, length, count)
        for curve, image in ((self.profile_ref, self.snapshot.pair.reference.frame.pixels), (self.profile_live, self.snapshot.pair.frame.pixels)):
            samples = cv2.remap(image, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[0]
            curve.setData(distance, samples)


class AnalyticsDock(QtWidgets.QDockWidget):
    changed = QtCore.Signal()
    capture_requested = QtCore.Signal()
    export_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__("Image analytics", parent)
        self.setObjectName("analyticsDock")
        self.setAllowedAreas(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea | QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        controls = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Off", "Live", "Frozen"])
        self.rate = QtWidgets.QDoubleSpinBox()
        self.rate.setRange(.1, 10)
        self.rate.setSingleStep(.5)
        self.rate.setValue(1)
        self.rate.setSuffix(" Hz")
        self.capture = QtWidgets.QPushButton("Capture pair")
        self.export = QtWidgets.QPushButton("Export inspection…")
        self.maximize = QtWidgets.QPushButton("Maximize")
        self.maximize.setToolTip("Expand analytics into a separate, freely resizable window")
        self.redock = QtWidgets.QPushButton("Dock")
        self.redock.setToolTip("Return analytics to the main window")
        self.redock.setVisible(self.isFloating() and parent is not None)
        controls.addWidget(QtWidgets.QLabel("Analytics"))
        controls.addWidget(self.mode)
        controls.addWidget(self.rate)
        controls.addWidget(self.capture)
        controls.addStretch()
        controls.addWidget(self.maximize)
        controls.addWidget(self.redock)
        controls.addWidget(self.export)
        layout.addLayout(controls)
        self.tabs = QtWidgets.QTabWidget()
        self.pages = {}
        for tab in ("Image", "Preprocessing", "Fourier", "Correlation", "Alignment", "Performance"):
            page = AnalysisPage(tab)
            self.pages[tab] = page
            self.tabs.addTab(page, tab)
            page.enabled.toggled.connect(self.changed)
        layout.addWidget(self.tabs)
        self.status = QtWidgets.QLabel("Off · live alignment keeps running")
        layout.addWidget(self.status)
        self.setWidget(content)
        self.mode.currentIndexChanged.connect(self.changed)
        self.rate.valueChanged.connect(self.changed)
        self.tabs.currentChanged.connect(self._tab_changed)
        self.capture.clicked.connect(self.capture_requested)
        self.export.clicked.connect(self.export_requested)
        self.maximize.clicked.connect(self._toggle_maximized)
        self.redock.clicked.connect(self._redock)
        self.topLevelChanged.connect(self.redock.setVisible)
        self.visibilityChanged.connect(self.changed)
        self._previous_tab = self.current_tab

    def _toggle_maximized(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.setFloating(True)
            self.showMaximized()

    def _redock(self) -> None:
        self.showNormal()
        self.setFloating(False)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.WindowStateChange:
            self.maximize.setText("Restore" if self.isMaximized() else "Maximize")

    @property
    def current_tab(self) -> str:
        return self.tabs.tabText(self.tabs.currentIndex())

    @property
    def active(self) -> bool:
        return self.isVisible() and self.mode.currentText() != "Off" and self.pages[self.current_tab].enabled.isChecked()

    def _tab_changed(self) -> None:
        if self._previous_tab in self.pages:
            self.pages[self._previous_tab].clear()
        self._previous_tab = self.current_tab
        self.changed.emit()

    def clear(self) -> None:
        for page in self.pages.values():
            page.clear()

    def display(self, snapshot: DiagnosticSnapshot) -> None:
        if snapshot.tab == self.current_tab:
            self.pages[snapshot.tab].display(snapshot)
