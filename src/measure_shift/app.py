"""Linux desktop application for live image alignment and optional inspection."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
import math
from pathlib import Path
import sys
import time

import numpy as np
# Load QtWidgets before QtGui (including pyqtgraph's imports). With a Conda
# interpreter, QtGui first can mix Conda/system xkbcommon libraries and crash
# inside QApplication's X11 keyboard setup.
from PySide6 import QtWidgets
from PySide6 import QtCore, QtGui
import pyqtgraph as pg

from .models import FramePacket, MatchConfig, ReferenceSnapshot, Region, new_id
from .motion import REFERENCE_CAPTURE_INTERVAL_S, MotionConfig, MotionContext, MotionWorker
from .motion_widgets import AxisKeyboardFilter, MotionControls, MotionDock
from .sources import create_source, discover_sources, read_gray
from .storage import export_bundle, export_csv, load_profile, measurement_row, save_profile
from .widgets import AnalyticsDock, ImagePane, MessageStatusBar
from .workers import AnalyticsWorker, BackgroundTask, CaptureWorker, LatestValue, MeasurementWorker


_READOUT_WINDOW_S = 0.15


def _make_preview(pair, mode):
    from .display import render_view
    started = time.perf_counter()
    image = render_view(pair, mode)
    return pair, mode, image, (time.perf_counter() - started) * 1000


def default_settings_path() -> Path:
    root = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.GenericConfigLocation)
    return Path(root) / "measure-shift" / "settings.json"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *, motion_hardware=None, settings_path=None):
        super().__init__()
        self._settings_path = Path(settings_path) if settings_path is not None else default_settings_path()
        self._restoring_settings = True
        self._remembered_camera = {}
        self.setWindowTitle("Measure Shift · Live alignment")
        self.resize(1500, 960)
        self.config = MatchConfig()
        self.reference = None
        self.latest_frame = None
        self._displayed_frame = None
        self.last_pair = None
        self._readout_samples = deque()
        self._readout_key = None
        self.last_diagnostic = None
        self.frozen_pair = None
        self.frames = LatestValue()
        self.capture_worker = None
        self.retired_captures = []
        self.measurement = MeasurementWorker(self.frames)
        self.analytics = AnalyticsWorker()
        self.preview = AnalyticsWorker(_make_preview)
        self.tasks = []
        self.history = deque(maxlen=10000)
        self.trend_history = deque(maxlen=600)
        self.started_at = time.monotonic()
        self._frame_revision = -1
        self._result_revision = -1
        self._analysis_revision = -1
        self._preview_revision = -1
        self._analysis_request = None
        self._preview_request = None
        self._last_analysis_submit = 0.0
        self._last_render = None
        self._last_preview_ms = 0.0
        self._capture_error = None
        self._parameter_error = None
        self._control_owner = None
        self._controls_snapshot = None
        self._pending_start = None
        self._auto_reference = False
        self._paused = False
        self._closing = False
        self._close_pending = False
        self._profile_camera = {}
        self._axis_source_kind = ""
        self._axis_source_id = ""
        self._motion_revision = -1
        self._motion_config_revision = 0
        self._motion_status = None
        self._reference_capture_request = None
        self._last_digital_reference_at = None
        self.motion = MotionWorker(hardware=motion_hardware, freshness_callback=self._request_motion_frames,
                                   calibration_worker=self.measurement)
        self._build_ui()
        self.axis_keyboard = AxisKeyboardFilter(self)
        self.axis_keyboard.changed.connect(self.motion.set_keyboard)
        QtWidgets.QApplication.instance().installEventFilter(self.axis_keyboard)
        self.settings_timer = QtCore.QTimer(self)
        self.settings_timer.setSingleShot(True)
        self.settings_timer.setInterval(350)
        self.settings_timer.timeout.connect(self._save_startup_settings)
        self._restore_startup_settings()
        self._connect_settings_autosave()
        self._restoring_settings = False
        self._connect_motion(True)
        self.motion.start()
        self.measurement.start()
        self.analytics.start()
        self.preview.start()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(33)

    def _build_ui(self):
        toolbar = self.main_toolbar = self.addToolBar("Main controls")
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.tools_menu = QtWidgets.QMenu("Tools", self)
        self.discover_action = self.tools_menu.addAction("Discover cameras", self.discover)
        self.tools_menu.addAction("Disconnect source", lambda: self.disconnect_source())
        self.tools_menu.addSeparator()
        self.tools_menu.addAction("Load reference…", self.load_reference)
        self.source_kind = QtWidgets.QComboBox()
        self.source_kind.addItems(["Demo", "IDS", "Basler", "Images", "Video"])
        self.device = QtWidgets.QComboBox()
        self.device.setMinimumWidth(140)
        self.device.setMaximumWidth(180)
        self.device.setMinimumContentsLength(14)
        self.device.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.device.addItem("Built-in moving target", "")
        toolbar.addWidget(self.source_kind)
        toolbar.addWidget(self.device)
        self.connect_button = QtWidgets.QPushButton("Connect / Open")
        toolbar.addWidget(self.connect_button)
        toolbar.addSeparator()
        toolbar.addAction("Capture reference", self.capture_reference)
        self.pause_action = self.tools_menu.addAction("Pause view")
        self.pause_action.setCheckable(True)
        self.pause_action.toggled.connect(self._pause)
        self.source_kind.currentTextChanged.connect(self._source_changed)
        self.connect_button.clicked.connect(self.connect_selected)

        toolbar.addSeparator()
        self.motion_controls = MotionControls(toolbar)
        tools_spacer = QtWidgets.QWidget()
        tools_spacer.setObjectName("toolbarSpacer")
        tools_spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        toolbar.addWidget(tools_spacer)
        toolbar.addSeparator()
        self.tools_button = QtWidgets.QToolButton()
        self.tools_button.setText("Tools")
        self.tools_button.setToolTip("Camera discovery, reference loading, view controls, snapshots, and settings")
        self.tools_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.tools_button.setMenu(self.tools_menu)
        toolbar.addWidget(self.tools_button)
        self.tools_menu.addAction("Save snapshot…", self.save_snapshot)
        self.tools_menu.addSeparator()

        content = QtWidgets.QWidget()
        main = QtWidgets.QVBoxLayout(content)
        heading = QtWidgets.QHBoxLayout()
        self.notice = QtWidgets.QLabel("Capture a reference, then adjust the part while watching the remaining offset.")
        self.notice.setWordWrap(False)
        self.notice.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self.notice.setObjectName("notice")
        heading.addWidget(self.notice, 1)
        self.connection_label = QtWidgets.QLabel("No source connected")
        self.connection_label.setObjectName("muted")
        heading.addWidget(self.connection_label)
        main.addLayout(heading)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.reference_pane = ImagePane("REFERENCE · capture or load an image")
        self.live_pane = ImagePane("LIVE · waiting for input")
        for pane in (self.reference_pane, self.live_pane):
            pane.plot.hideAxis("bottom")
            pane.plot.hideAxis("left")
            pane.cursor.hide()
        self.reference_pane.view.sigRangeChangedManually.connect(lambda _: self._link_views(self.reference_pane, self.live_pane))
        self.live_pane.view.sigRangeChangedManually.connect(lambda _: self._link_views(self.live_pane, self.reference_pane))
        split.addWidget(self.reference_pane)
        split.addWidget(self.live_pane)
        self.reference_pane.region_changed.connect(self._region_dragged)
        self.live_pane.region_changed.connect(self._region_dragged)

        sidebar = QtWidgets.QWidget()
        sidebar.setMinimumWidth(245)
        sidebar.setMaximumWidth(310)
        side = QtWidgets.QVBoxLayout(sidebar)
        self.state = QtWidgets.QLabel("NO REFERENCE")
        self.state.setObjectName("state")
        self.state.setWordWrap(True)
        side.addWidget(self.state)
        self.x_value = QtWidgets.QLabel("X   — px")
        self.y_value = QtWidgets.QLabel("Y   — px")
        for label in (self.x_value, self.y_value):
            label.setObjectName("readout")
            label.setToolTip("Display averaged over 150 ms; tolerance uses the latest measurement.")
            side.addWidget(label)
        form = QtWidgets.QFormLayout()
        self.tolerance = QtWidgets.QDoubleSpinBox()
        self.tolerance.setRange(0, 100000)
        self.tolerance.setDecimals(2)
        self.tolerance.setValue(1)
        self.tolerance.setSuffix(" px")
        form.addRow("Tolerance", self.tolerance)
        self.area = QtWidgets.QComboBox()
        self.area.addItems(["Whole image", "Rectangle", "Circle"])
        form.addRow("Analyze", self.area)
        self.scale = QtWidgets.QComboBox()
        self.scale.addItem("Native (1:1)", 1.0)
        self.scale.addItem("Half (1:2)", .5)
        self.scale.addItem("Quarter (1:4)", .25)
        form.addRow("Resolution", self.scale)
        self.view_mode = QtWidgets.QComboBox()
        self.view_mode.addItems(["Live", "Red / cyan overlay", "Absolute difference", "Compensated preview", "Compensated difference"])
        form.addRow("View", self.view_mode)
        side.addLayout(form)
        self.crosshair_toggle = QtWidgets.QCheckBox("Show center crosshair")
        self.crosshair_toggle.setToolTip("Show fixed guides through the image center in both reference and live views.")
        side.addWidget(self.crosshair_toggle)
        self.crosshair_toggle.toggled.connect(self.reference_pane.set_crosshair_visible)
        self.crosshair_toggle.toggled.connect(self.live_pane.set_crosshair_visible)
        self.tolerance.valueChanged.connect(self._update_readout)
        self.area.currentIndexChanged.connect(self._area_changed)
        self.scale.currentIndexChanged.connect(self._matching_changed)
        self.view_mode.currentIndexChanged.connect(self._view_changed)

        camera = QtWidgets.QGroupBox("Camera controls")
        camera_layout = QtWidgets.QFormLayout(camera)
        self.camera_controls = {}
        pixel_clock = QtWidgets.QComboBox()
        pixel_clock.setPlaceholderText("Connect an IDS camera")
        pixel_clock.setEnabled(False)
        pixel_clock.setToolTip("Apply pixel clock changes to refresh the frame rate and exposure limits.")
        camera_layout.addRow("Pixel clock (MHz)", pixel_clock)
        self.camera_controls["pixel_clock_mhz"] = pixel_clock
        for key, text in (("fps", "Frame rate (fps)"), ("exposure_ms", "Exposure (ms)"), ("gain", "Gain")):
            control = QtWidgets.QDoubleSpinBox()
            control.setDecimals(6 if key == "exposure_ms" else 3)
            control.setRange(0, 1000000)
            control.setEnabled(False)
            camera_layout.addRow(text, control)
            self.camera_controls[key] = control
        self.apply_camera = QtWidgets.QPushButton("Apply camera settings")
        camera_layout.addRow(self.apply_camera)
        self.apply_camera.clicked.connect(self._apply_camera)
        side.addWidget(camera)
        side.addStretch()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(sidebar)
        scroll.setMinimumWidth(270)
        scroll.setMaximumWidth(335)
        split.addWidget(scroll)
        split.setSizes([560, 560, 290])
        main.addWidget(split, 1)

        self.trend_plot = pg.PlotWidget()
        self.trend_plot.setMaximumHeight(160)
        self.trend_plot.setLabel("left", "Displacement", units="px")
        self.trend_plot.setLabel("bottom", "Time since start", units="s")
        self.trend_plot.addLegend()
        self.trend_x = self.trend_plot.plot(name="X", pen=pg.mkPen("#ff7979", width=2))
        self.trend_y = self.trend_plot.plot(name="Y", pen=pg.mkPen("#62d4ef", width=2))
        self.trend_plot.hide()
        main.addWidget(self.trend_plot)
        self.performance_bar = QtWidgets.QWidget()
        performance = QtWidgets.QHBoxLayout(self.performance_bar)
        performance.setContentsMargins(0, 0, 0, 0)
        performance.setSpacing(16)
        self.performance_labels = {}
        for title in ("Camera FPS", "Measurement FPS", "Processing", "Scale", "Analytics"):
            label = QtWidgets.QLabel(f"{title} —")
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
            self.performance_labels[title] = label
            performance.addWidget(label)
        status_bar = MessageStatusBar(self.motion_controls.status, self)
        self.setStatusBar(status_bar)
        status_area = QtWidgets.QWidget()
        status_area.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        status_layout = QtWidgets.QHBoxLayout(status_area)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(6)
        status_layout.addWidget(status_bar.message_area, 1)
        performance.insertWidget(2, status_area, 1)
        status_bar.addPermanentWidget(self.performance_bar, 1)
        self.setCentralWidget(content)
        self.dock = AnalyticsDock(self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.dock)
        self.dock.hide()
        self.tools_menu.addAction(self.dock.toggleViewAction())
        self.dock.changed.connect(self._analysis_changed)
        self.dock.capture_requested.connect(self.capture_inspection)
        self.dock.export_requested.connect(self.export_inspection)
        self.motion_dock = MotionDock(parent=self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.motion_dock)
        self.motion_dock.hide()
        self.tools_menu.addAction(self.motion_dock.toggleViewAction())
        self.motion_dock.config_changed.connect(self._motion_config_changed)
        self.motion_controls.connect_button.toggled.connect(self._connect_motion)
        self.motion_dock.setup_requested.connect(self.motion.apply_setup)
        self.motion_dock.arm_requested.connect(self.motion.arm)
        self.motion_controls.stop_button.clicked.connect(self._stop_motion)
        self.motion_controls.calibrate_button.clicked.connect(
            lambda: self._calibrate_motion(self.motion_dock.test_steps.value()))
        self.motion_dock.test_steps.valueChanged.connect(self.motion_controls.set_calibration_move)
        self.motion_dock.known_distance_requested.connect(self._set_known_pixel_distance)
        self.motion_controls.align_button.clicked.connect(self._align_motion)
        self.motion_dock.error.connect(self._error)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction("Open current image…", self.open_image)
        file_menu.addAction("Open image sequence folder…", self.open_sequence)
        file_menu.addAction("Open recorded video…", self.open_video)
        file_menu.addAction("Load reference…", self.load_reference)
        file_menu.addSeparator()
        file_menu.addAction("Save snapshot bundle…", self.save_snapshot)
        file_menu.addAction("Export CSV history (last 10,000)…", self.save_csv)
        file_menu.addAction("Save settings profile…", self.save_settings)
        file_menu.addAction("Load settings profile…", self.load_settings)
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close)
        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.dock.toggleViewAction())
        view_menu.addAction(self.motion_dock.toggleViewAction())
        self.trend_toggle = view_menu.addAction("Show offset history (last 600 measurements)")
        self.trend_toggle.setCheckable(True)
        self.trend_toggle.toggled.connect(self.trend_plot.setVisible)
        self.trend_toggle.toggled.connect(self._draw_trend)
        view_menu.addAction("Fit images", self._fit_images)
        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction("About / workflow", self._about)
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #19232e; color: #d7e2eb; font-size: 12px; }
            QToolBar, QMenuBar { background: #202e3b; spacing: 4px; padding: 5px; }
            QPushButton, QComboBox, QDoubleSpinBox { background: #293a49; border: 1px solid #456071; border-radius: 4px; padding: 5px; }
            QPushButton:hover { background: #355367; } QPushButton:disabled { color: #73818d; }
            QToolBar::separator { background: #456071; width: 1px; margin: 4px; }
            QWidget#toolbarSpacer { background: transparent; }
            QToolBar QToolButton { background: #293a49; border: 1px solid #456071; border-radius: 4px; padding: 5px; }
            QToolBar QToolButton:hover { background: #355367; }
            QToolBar QToolButton:checked { background: #416174; }
            QPushButton#axisConnectButton { border-color: #568a91; }
            QPushButton#axisConnectButton:!checked { background: #416174; border-color: #79bac2; }
            QPushButton#alignXButton { background: #285344; border-color: #5d9d83; }
            QPushButton#alignXButton:hover { background: #326a55; }
            QPushButton#axisStopButton { background: #773e38; border-color: #b86c62; color: #fff1ee; font-weight: bold; }
            QPushButton#axisStopButton:hover { background: #944a42; }
            QPushButton#alignXButton:disabled { background: #253d35; border-color: #456958; color: #81958b; }
            QPushButton#axisStopButton:disabled { background: #4b302e; border-color: #774c46; color: #a38b85; }
            QGroupBox { border: 1px solid #3a4b5a; border-radius: 5px; margin-top: 12px; padding-top: 12px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #9eb9ca; }
            QLabel#appTitle { font-size: 22px; font-weight: 700; letter-spacing: 2px; color: #69dfb7; }
            QLabel#paneTitle { font-size: 13px; font-weight: 600; padding: 5px; }
            QLabel#readout { font-size: 24px; font-weight: 600; padding: 3px; }
            QLabel#state { font-size: 15px; font-weight: 700; padding: 8px; border-radius: 4px; background: #293a49; }
            QLabel#notice { color: #b7cbd8; padding: 5px; }
            QLabel#muted { color: #99adbc; font-size: 11px; }
            QTabWidget::pane { border: 1px solid #456071; }
            QTabBar::tab { background: #293a49; padding: 7px; } QTabBar::tab:selected { background: #456071; }
            QCheckBox { padding: 3px; } QScrollArea { border: 0; }
            QStatusBar::item { border: 0; }
        """)
        # Reserve a single row; changing digits and messages must not move the views.
        self.performance_bar.ensurePolished()
        line_height = self.performance_bar.fontMetrics().lineSpacing()
        self.performance_bar.setFixedHeight(line_height + 4)
        for title, sample in (("Camera FPS", "999999.9"), ("Measurement FPS", "999999.9"),
                              ("Processing", "999999.9 ms"), ("Scale", "0.25"), ("Analytics", "frozen")):
            label = self.performance_labels[title]
            metrics = label.fontMetrics()
            label.setFixedSize(metrics.horizontalAdvance(f"{title} {sample}") + 8, line_height)
        self.motion_controls.status.setFixedHeight(line_height)
        # Reserve the longer label after the toolbar font and style are applied.
        hardware_button = self.motion_controls.connect_button
        hardware_button.setText("Disconnect hardware")
        hardware_button.ensurePolished()
        hardware_button.setFixedWidth(hardware_button.sizeHint().width())
        hardware_button.setText("Connect hardware")
        self._source_changed(self.source_kind.currentText())

    def _task(self, function, callback, *, finish_before_close=False):
        task = BackgroundTask(function, callback, self._error)
        task.finish_before_close = finish_before_close
        self.tasks.append(task)
        task.start()

    def _error(self, error):
        self.notice.setText(str(error))
        self.statusBar().showMessage(str(error), 15000)

    def _request_motion_frames(self, after):
        """Called by the motion worker; only the capture mailbox is touched."""
        capture = self.capture_worker
        if capture is None or not capture.is_alive() or capture.error:
            raise RuntimeError("Live camera disconnected while waiting for motion feedback")
        return capture.request_fresh_frames(after)

    def _axis_context(self):
        capture = self.capture_worker
        connected = bool(capture and capture.is_alive() and capture.status == "Connected" and not capture.error)
        fps = capture.controls.get("fps", {}).get("value", 30) if capture else 30
        total_ms = self.last_pair.result.timings.get("total_ms", 0) if self.last_pair else 0
        exposure_ms = capture.controls.get("exposure_ms", {}).get("value", 0) if capture else 0
        identity = getattr(capture.source, "device_identity", None) if connected else None
        source_id = f"{self._axis_source_kind}:serial:{identity}" if identity else self._axis_source_id
        return MotionContext(source_id=source_id, source_kind=self._axis_source_kind,
                             shape=self.latest_frame.shape if self.latest_frame is not None else None,
                             reference_id=self.reference.reference_id if self.reference else None,
                             config_id=self.config.config_id, connected=connected, paused=self._paused,
                             tolerance_px=self.tolerance.value(),
                             settling_s=max(.2, exposure_ms / 1000 + 1 / max(.1, fps)),
                             max_feedback_age_s=max(1., 3 / max(.1, fps), 3 * total_ms / 1000))

    def _motion_config_changed(self, config):
        self._motion_config_revision = self.motion.configure(config)
        self._schedule_settings_save()

    def _connect_motion(self, connected):
        self.axis_keyboard.clear()
        if connected:
            revision = self.motion.connect_hardware()
        else:
            revision = self.motion.disconnect_hardware()
        self.motion_controls.set_connection_pending(connected, revision)

    def _stop_motion(self):
        self.axis_keyboard.clear()
        self.motion.stop()

    def _calibrate_motion(self, steps):
        self.motion.set_context(self._axis_context())
        try:
            self.motion.calibrate(steps)
        except ValueError as exc:
            self._error(exc)

    def _align_motion(self):
        self.motion.set_context(self._axis_context())
        self.motion.align()

    def _set_known_pixel_distance(self, pixels):
        context = self._axis_context()
        if not context.live_camera_ready:
            self._error("Connect a live IDS or Basler camera before entering known pixel movement")
            return
        try:
            if not math.isfinite(pixels) or pixels == 0:
                raise ValueError("Known pixel movement must be finite and nonzero")
            old = self.motion_dock.config
            scale = pixels / old.distance_steps()
            config = replace(old, pixels_per_step=scale,
                             calibration_source_id=context.source_id, calibration_shape=context.shape,
                             jog_unit="pixels", jog_distance=abs(pixels)).validated()
        except ValueError as exc:
            self._error(exc)
            return
        self.motion.set_context(context)
        self.motion_dock.set_config(config, preserve_input_edits=True)
        self._motion_config_changed(config)
        self.notice.setText(f"Known pixel distance saved: {pixels:+.6g} px/nudge ({scale:+.6g} px/driver step)")

    def _poll_motion(self):
        context = self._axis_context()
        self.motion.set_context(context)
        status, revision = self.motion.output.read()
        if status is None:
            return
        self._motion_status = status
        if revision != self._motion_revision:
            self._motion_revision = revision
            if status.config_revision == self._motion_config_revision and status.config != self.motion_dock.config:
                self.motion_dock.set_config(status.config, preserve_input_edits=True)
                self._schedule_settings_save()
        self.motion_dock.display_status(status, context)
        self.motion_controls.display_status(status, context)
        request = status.reference_capture
        if request != self._reference_capture_request:
            self._reference_capture_request = request
            if (request is not None and status.connected
                    and request.config_revision == self._motion_config_revision
                    and request.channel == self.motion_dock.config.reference_channel):
                now = time.monotonic()
                # GUI stalls can bring separately sampled edges close together
                # when handled. Bound actual captures as well as input requests.
                if (self._last_digital_reference_at is None or
                        now - self._last_digital_reference_at >= REFERENCE_CAPTURE_INTERVAL_S):
                    self._last_digital_reference_at = now
                    self.capture_reference()

    def _source_changed(self, kind):
        self.device.clear()
        self.device.addItem("Discover cameras" if kind in ("IDS", "Basler") else "Built-in moving target" if kind == "Demo" else "Choose a file or folder…", "")
        self.discover_action.setEnabled(kind in ("IDS", "Basler"))

    def discover(self):
        kind = self.source_kind.currentText()
        self.discover_action.setEnabled(False)
        def done(devices):
            self.discover_action.setEnabled(True)
            if kind != self.source_kind.currentText():
                return
            self.device.clear()
            for device in devices:
                self.device.addItem(device["label"], device["id"])
            if not devices:
                self.device.addItem("No cameras found", "")
                self.notice.setText(f"No {kind} cameras found. Connect the camera and check it in the manufacturer's viewer.")
        def action():
            return discover_sources(kind)
        task = BackgroundTask(action, done, lambda error: (self.discover_action.setEnabled(True), self._error(error)))
        self.tasks.append(task)
        task.start()

    def connect_selected(self):
        kind = self.source_kind.currentText()
        if kind == "Images":
            self.open_image()
        elif kind == "Video":
            self.open_video()
        else:
            self.start_source(kind, identifier=self.device.currentData() or "", auto_reference=kind == "Demo")

    def start_source(self, kind, identifier="", path="", auto_reference=False, preserve_reference=False):
        self.disconnect_source(clear_reference=not preserve_reference)
        request = kind, identifier, path, auto_reference
        if any(worker.is_alive() for worker in self.retired_captures):
            self._pending_start = request
            self.notice.setText("Closing the previous input before opening the selected source…")
        else:
            self._open_source(*request)

    def _open_source(self, kind, identifier, path, auto_reference):
        display_kind = kind.capitalize() if kind.lower() != "ids" else "IDS"
        kind_changed = self.source_kind.currentText() != display_kind
        self.source_kind.blockSignals(True)
        self.source_kind.setCurrentText(display_kind)
        self.source_kind.blockSignals(False)
        if kind_changed or kind.lower() in ("demo", "images", "video"):
            self._source_changed(display_kind)
        if kind.lower() in ("images", "video"):
            self.device.clear()
            self.device.addItem(Path(path).name or "Selected file", path)
        elif identifier:
            index = self.device.findData(identifier)
            if index < 0:
                self.device.addItem(identifier, identifier)
                index = self.device.count() - 1
            self.device.setCurrentIndex(index)
        try:
            source = create_source(kind, identifier=identifier, path=path)
            self._axis_source_kind = kind.lower()
            self._axis_source_id = f"{kind.lower()}:session:{new_id()}"
            maximize_timing = kind.lower() in ("ids", "basler")
            if maximize_timing:
                # Startup timing comes from the camera's current limits, not
                # previously saved values staged in the controls.
                for key in ("pixel_clock_mhz", "fps", "exposure_ms"):
                    self._profile_camera.pop(key, None)
            self.capture_worker = CaptureWorker(source, self.frames, maximize_timing=maximize_timing)
            self._auto_reference = auto_reference
            self._capture_error = None
            self._parameter_error = None
            self._control_owner = None
            self.capture_worker.start()
            self.notice.setText("Starting input… Capture or load a reference when the image is ready.")
        except Exception as exc:
            self._error(exc)

    def disconnect_source(self, clear_reference=True):
        self._settings_camera()
        self._stop_motion()
        self._axis_source_kind = self._axis_source_id = ""
        self._pending_start = None
        if self.capture_worker is not None:
            self.capture_worker.stop()
            self.retired_captures.append(self.capture_worker)
            self.capture_worker = None
        self.frames.clear()
        self.config = replace(self.config, live_region=None, config_id=new_id())
        self.last_pair = None
        self.measurement.configure(self.reference, self.config)
        self.latest_frame = None
        self._displayed_frame = None
        self._auto_reference = False
        self.preview.cancel()
        self._preview_request = None
        for control in self.camera_controls.values():
            control.setEnabled(False)
        if clear_reference:
            self._clear_reference("Disconnected · connect an input to continue")
        else:
            self._analysis_changed()
        self._update_readout()

    def capture_reference(self):
        frame = self._displayed_frame if self._paused else self.latest_frame
        if frame is None:
            self._error("No current image available. Connect a camera, start Demo, or open an image first.")
            return
        self.set_reference(frame)

    def set_reference(self, frame):
        self._stop_motion()
        self.reference = ReferenceSnapshot(frame)
        try:
            self.config.region.bounds(frame.shape)
        except ValueError:
            self.config = replace(self.config, region=Region(), live_region=None, config_id=new_id())
            self._sync_matching_controls()
        self.last_pair = None
        self.reference_pane.set_image(frame.pixels, auto=True)
        self.reference_pane.title.setText(f"REFERENCE · {frame.source_name} · {frame.shape[1]} × {frame.shape[0]}")
        self.reference_pane.clear_annotations()
        self.live_pane.clear_annotations()
        self.measurement.configure(self.reference, self.config)
        self.reference_pane.set_region(self.config.region)
        self._analysis_changed()
        self._view_changed()
        self._update_readout()
        self.notice.setText("Reference captured. Adjust the part and watch X / Y displacement. Yellow arrows show image correction.")

    def _clear_reference(self, reason):
        self._stop_motion()
        self.reference = None
        self.last_pair = None
        self.measurement.configure(None, self.config)
        self.reference_pane.clear_image()
        self.reference_pane._region = None
        self.reference_pane.set_region(Region())
        self.reference_pane.clear_annotations()
        self.live_pane.clear_annotations()
        self.reference_pane.title.setText("REFERENCE · capture or load an image")
        self.notice.setText(reason)
        self._analysis_changed()
        self._view_changed()
        self._update_readout()
        self.live_pane.show_correction(None, None)

    def _choose_file(self, title, filters):
        return QtWidgets.QFileDialog.getOpenFileName(self, title, str(Path.cwd() / "images"), filters)[0]

    def load_reference(self):
        path = self._choose_file("Load reference image", "Images (*.bmp *.png *.tif *.tiff *.jpg *.jpeg);;All files (*)")
        if path:
            self._task(lambda: FramePacket.from_array(read_gray(path), Path(path).name), self.set_reference)

    def open_image(self):
        path = self._choose_file("Open current image", "Images (*.bmp *.png *.tif *.tiff *.jpg *.jpeg);;All files (*)")
        if path:
            self.start_source("Images", path=path, preserve_reference=True)

    def open_sequence(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open image sequence folder", str(Path.cwd() / "images"))
        if path:
            self.start_source("Images", path=path, preserve_reference=True)

    def open_video(self):
        path = self._choose_file("Open recorded video", "Video (*.mp4 *.avi *.mkv *.mov *.m4v);;All files (*)")
        if path:
            self.start_source("Video", path=path)

    def _default_region(self, kind):
        frame = self.reference.frame if self.reference else self.latest_frame
        if frame is None:
            return Region()
        h, w = frame.shape
        if kind == "circle":
            size = max(8, int(min(w, h) * .8))
            return Region(kind, (w - size) // 2, (h - size) // 2, size, size)
        return Region(kind, w // 4, h // 4, max(8, w // 2), max(8, h // 2))

    def _set_config(self, config):
        self._stop_motion()
        self.config = replace(config, live_region=None, config_id=new_id())
        self.last_pair = None
        self.measurement.configure(self.reference, self.config)
        self.reference_pane.set_region(self.config.region)
        self.live_pane.set_region(self.config.region)
        self._analysis_changed()
        self._view_changed()
        self._update_readout()
        self._schedule_settings_save()

    def _sync_matching_controls(self):
        for widget in (self.area, self.scale):
            widget.blockSignals(True)
        self.area.setCurrentIndex({"full": 0, "rectangle": 1, "circle": 2}[self.config.region.kind])
        self.scale.setCurrentIndex({1: 0, .5: 1, .25: 2}[self.config.scale])
        for widget in (self.area, self.scale):
            widget.blockSignals(False)

    def _area_changed(self):
        kind = ("full", "rectangle", "circle")[self.area.currentIndex()]
        region = Region() if kind == "full" else self._default_region(kind)
        self._set_config(replace(self.config, region=region))
        self._sync_matching_controls()

    def _matching_changed(self):
        self._set_config(replace(self.config, scale=self.scale.currentData()))

    def _region_dragged(self, region):
        self._set_config(replace(self.config, region=region))

    def _pause(self, paused):
        if paused:
            self._stop_motion()
        self._paused = paused
        if paused:
            self.measurement.paused.set()
        else:
            self.measurement.paused.clear()
            self._last_render = None
        self.pause_action.setText("Resume view" if paused else "Pause view")
        self._update_readout()

    def _view_changed(self):
        self.preview.cancel()
        self._preview_request = None
        self._last_render = None

    def _camera_values(self):
        values = {}
        for key, control in self.camera_controls.items():
            if control.isEnabled():
                value = control.currentData() if isinstance(control, QtWidgets.QComboBox) else control.value()
                if value is not None:
                    values[key] = value
        return values

    def _apply_camera(self):
        if self.capture_worker is None:
            return
        parameters = {}
        available = self.capture_worker.controls
        for key, value in self._camera_values().items():
            info = available.get(key)
            if info is None:
                continue
            control = self.camera_controls[key]
            tolerance = 0 if isinstance(control, QtWidgets.QComboBox) else 0.5 * 10 ** -control.decimals()
            # Avoid re-sending rounded readback values. The source preserves
            # unchanged timing within any new clock/rate-dependent limits.
            if not math.isclose(value, info["value"], abs_tol=tolerance):
                parameters[key] = min(info["max"], max(info["min"], value))
        if parameters:
            self._stop_motion()
            self.capture_worker.set_parameters(**parameters)

    def _install_controls(self, worker):
        for key, control in self.camera_controls.items():
            info = worker.controls.get(key)
            control.setEnabled(info is not None)
            if isinstance(control, QtWidgets.QComboBox):
                control.clear()
                if info:
                    control.setPlaceholderText("Select a supported clock")
                    for value in info["values"]:
                        control.addItem(f"{value:g}", value)
                    requested = self._profile_camera.get(key, info["value"])
                    index = control.findData(requested)
                    control.setCurrentIndex(index if index >= 0 else control.findData(info["value"]))
                else:
                    control.setPlaceholderText("Connect an IDS camera")
                continue
            if info:
                control.setRange(float(info["min"]), float(info["max"]))
                control.setSingleStep(max(10 ** -control.decimals(), float(info.get("step", 1))))
                control.setValue(float(self._profile_camera.get(key, info["value"])))
        self._control_owner = worker
        self._controls_snapshot = worker.controls
        self._profile_camera = {}

    def _analysis_changed(self):
        self.analytics.cancel()
        self.last_diagnostic = None
        self._analysis_request = None
        self._last_analysis_submit = 0
        self.dock.clear()
        mode = self.dock.mode.currentText()
        if mode == "Off":
            self.frozen_pair = None
            self.dock.status.setText("Off · diagnostic work stopped; live alignment continues")
        elif mode == "Frozen" and self.frozen_pair is None:
            self.frozen_pair = self.last_pair
        if not self.dock.isVisible() and mode != "Off":
            self.dock.status.setText("Dock hidden · live diagnostic updates paused")

    def capture_inspection(self):
        if self.last_pair is None:
            self._error("Capture a reference and wait for a measurement before freezing an inspection pair.")
            return
        self.frozen_pair = self.last_pair
        self.dock.mode.blockSignals(True)
        self.dock.mode.setCurrentText("Frozen")
        self.dock.mode.blockSignals(False)
        self.dock.show()
        self._analysis_changed()

    def _schedule_analytics(self):
        if not self.dock.active:
            return
        mode = self.dock.mode.currentText()
        pair = self.frozen_pair if mode == "Frozen" else self.last_pair
        if pair is None:
            self.dock.status.setText("Waiting for a measured reference/frame pair")
            return
        key = pair.key, self.dock.current_tab
        if key == self._analysis_request:
            return
        now = time.monotonic()
        if mode == "Live" and now - self._last_analysis_submit < 1 / self.dock.rate.value():
            return
        self.analytics.submit(pair, self.dock.current_tab)
        self._analysis_request = key
        self._last_analysis_submit = now
        self.dock.status.setText(f"{mode} · computing {self.dock.current_tab} for frame {pair.frame.frame_id[:8]}…")

    def _update_readout(self):
        pair = self.last_pair
        result = pair.result if pair else None
        if self._paused:
            state, color = "PAUSED", "#ffcd66"
        elif self.capture_worker is None or self.capture_worker.error:
            state, color = "DISCONNECTED", "#ff7979"
        elif self.capture_worker.status == "End of input":
            state, color = "INPUT ENDED", "#ffcd66"
        elif self.reference is None:
            state, color = "NO REFERENCE", "#9eb9ca"
        elif self._motion_status is not None and self._motion_status.phase == "aligning":
            state, color = "ALIGNING", "#9eb9ca"
        elif result is None:
            state, color = "MEASURING…", "#9eb9ca"
        elif time.monotonic() - pair.frame.received_at > max(1.0,
                3 / max(.1, self.capture_worker.controls.get("fps", {}).get("value", 30)),
                3 * result.timings.get("total_ms", 0) / 1000):
            state, color = "STALE MEASUREMENT", "#ff7979"
        elif not result.usable:
            state, color = "MATCH UNCERTAIN", "#ffcd66"
        elif result.distance <= self.tolerance.value():
            state, color = "ALIGNED", "#69dfb7"
        else:
            state, color = "OUTSIDE TOLERANCE", "#ffcd66"
        self.state.setText(state)
        if getattr(self, "_state_color", None) != color:
            self.state.setStyleSheet(f"color: {color}")
            self._state_color = color
        finite = result is not None and np.isfinite([result.dx, result.dy]).all()
        if result is None or not result.usable:
            self._readout_samples.clear()
            self._readout_key = None
        elif pair.key != self._readout_key:
            # Count each measured frame once, even when the UI refreshes more often.
            if self._readout_key is None or pair.key[0::2] != self._readout_key[0::2]:
                self._readout_samples.clear()
            now = time.monotonic()
            while self._readout_samples and now - self._readout_samples[0][0] > _READOUT_WINDOW_S:
                self._readout_samples.popleft()
            self._readout_samples.append((now, result.dx, result.dy))
            self._readout_key = pair.key
        if finite:
            dx, dy = result.dx, result.dy
            if self._readout_samples:
                count = len(self._readout_samples)
                dx = sum(sample[1] for sample in self._readout_samples) / count
                dy = sum(sample[2] for sample in self._readout_samples) / count
            # The z format option avoids a flickering minus sign on rounded zero.
            self.x_value.setText(f"X   {dx:+z.1f} px")
            self.y_value.setText(f"Y   {dy:+z.1f} px")
        else:
            self.x_value.setText("X   — px")
            self.y_value.setText("Y   — px")

    def _render(self):
        if self._paused or self.latest_frame is None:
            return
        mode = self.view_mode.currentText()
        pair = self.last_pair
        if mode == "Live":
            frame = self.latest_frame
            key = (mode, frame.frame_id, self.config.config_id)
            if key != self._last_render:
                self.live_pane.set_image(frame.pixels)
                self._displayed_frame = frame
                self.live_pane.title.setText(f"LIVE · {frame.source_name} · {frame.shape[1]} × {frame.shape[0]}")
                self._last_render = key
        elif pair is not None:
            key = pair.key, mode
            if self._preview_request != key:
                self.preview.submit(pair, mode)
                self._preview_request = key
        else:
            self.live_pane.set_image(self.latest_frame.pixels)
            self.live_pane.title.setText("LIVE · waiting for matching result to build selected view")
        self.reference_pane.set_region(self.config.region)
        self.live_pane.set_region(self.config.region)
        result = pair.result if pair else None
        self.live_pane.show_correction(result.dx if result and result.usable else None, result.dy if result and result.usable else None)
        self.live_pane.clear_annotations()

    def _draw_trend(self):
        if not self.trend_plot.isVisible() or not self.trend_history:
            return
        data = np.asarray(self.trend_history)
        self.trend_x.setData(data[:, 0], data[:, 1])
        self.trend_y.setData(data[:, 0], data[:, 2])

    def _poll(self):
        if self._closing:
            return
        for task in list(self.tasks):
            if not task.is_alive():
                self.tasks.remove(task)
                if task.error:
                    if getattr(task, "finish_before_close", False):
                        self._close_pending = False
                    task.error_callback(task.error)
                else:
                    task.callback(task.result)
        worker = self.capture_worker
        if worker:
            self.connection_label.setText(f"{worker.source.name} · {worker.status}")
            if worker.status == "Connected" and (self._control_owner is not worker or self._controls_snapshot is not worker.controls):
                self._install_controls(worker)
            if not worker.is_alive() and worker.status in ("Disconnected", "End of input"):
                for control in self.camera_controls.values():
                    control.setEnabled(False)
            if worker.error and worker.error != self._capture_error:
                self._capture_error = worker.error
                self._error(worker.error)
                self._update_readout()
            if worker.parameter_error and worker.parameter_error != self._parameter_error:
                self._parameter_error = worker.parameter_error
                self._error(worker.parameter_error)
            elif worker.parameter_error is None:
                self._parameter_error = None
        else:
            self.connection_label.setText("No source connected")
        frame, revision = self.frames.read()
        if revision != self._frame_revision:
            self._frame_revision = revision
            if frame is not None:
                self.latest_frame = frame
                if self.reference is not None and self.reference.frame.shape != frame.shape:
                    self._clear_reference("Image geometry changed. Capture a new reference before measuring.")
                    self.config = replace(self.config, region=Region(), live_region=None, config_id=new_id())
                    self._sync_matching_controls()
                if self._auto_reference:
                    self._auto_reference = False
                    self.set_reference(frame)
        pair, revision = self.measurement.output.read()
        if revision != self._result_revision:
            self._result_revision = revision
            if (not self._paused and pair is not None and self.reference is not None
                    and pair.key[0] == self.reference.reference_id and pair.key[2] == self.config.config_id):
                pair = replace(pair, runtime_metrics={
                    **pair.runtime_metrics,
                    "camera_fps": worker.rate.value if worker else 0,
                    "measurement_fps": self.measurement.rate.value,
                    "frame_age_at_display_ms": max(0, (time.monotonic() - pair.frame.received_at) * 1000),
                    "qt_image_paint_ms": self.reference_pane.plot.last_paint_ms + self.live_pane.plot.last_paint_ms,
                    "preview_ms": self._last_preview_ms,
                })
                self.last_pair = pair
                self.motion.submit_feedback(pair)
                self.history.append(measurement_row(pair))
                self.trend_history.append((time.monotonic() - self.started_at, pair.result.dx if pair.result.usable else math.nan,
                                           pair.result.dy if pair.result.usable else math.nan))
                self._update_readout()
                self._draw_trend()
        if self.measurement.error:
            self._error(self.measurement.error)
        preview, revision = self.preview.output.read()
        if revision != self._preview_revision:
            self._preview_revision = revision
            if preview and preview.generation == self.preview.generation and not preview.error and not self._paused:
                preview_pair, mode, image, preview_ms = preview.snapshot
                key = preview_pair.key
                if self.last_pair and key[0::2] == self.last_pair.key[0::2] and mode == self.view_mode.currentText():
                    self.live_pane.set_image(image)
                    self._last_preview_ms = preview_ms
                    self._displayed_frame = preview_pair.frame
                    self.live_pane.title.setText(f"{mode.upper()} · measured frame {key[1][:8]}")
        analysis, revision = self.analytics.output.read()
        if revision != self._analysis_revision:
            self._analysis_revision = revision
            if analysis and analysis.generation == self.analytics.generation and self.dock.active:
                if analysis.error:
                    self.dock.status.setText(analysis.error)
                else:
                    snapshot = analysis.snapshot
                    expected = self.frozen_pair if self.dock.mode.currentText() == "Frozen" else self.last_pair
                    # In live mode the camera may advance during diagnostics; the
                    # result remains useful when its reference/settings still match.
                    compatible = expected is not None and (
                        snapshot.pair.key == expected.key if self.dock.mode.currentText() == "Frozen"
                        else snapshot.pair.key[0::2] == expected.key[0::2])
                    if compatible and snapshot.tab == self.dock.current_tab:
                        self.last_diagnostic = snapshot
                        self.dock.display(snapshot)
                        self.dock.status.setText(f"{self.dock.mode.currentText()} · frame {snapshot.pair.frame.frame_id[:8]} · analytics {snapshot.metrics.get('analytics_ms', 0):.1f} ms")
        self._render()
        self._schedule_analytics()
        pair = self.last_pair
        camera_fps = worker.rate.value if worker and worker.is_alive() else 0
        processing = pair.result.timings.get("total_ms", 0) if pair else 0
        measurement_fps = 0 if self._paused else self.measurement.rate.value
        for title, value in (("Camera FPS", f"{camera_fps:.1f}"), ("Measurement FPS", f"{measurement_fps:.1f}"),
                             ("Processing", f"{processing:.1f} ms"), ("Scale", f"{self.config.scale:g}"),
                             ("Analytics", "busy" if self.analytics.busy else self.dock.mode.currentText().lower())):
            self.performance_labels[title].setText(f"{title} {value}")
        self._poll_motion()
        self._update_readout()
        self.retired_captures = [thread for thread in self.retired_captures if thread.is_alive()]
        if self._pending_start and not self.retired_captures:
            request, self._pending_start = self._pending_start, None
            self._open_source(*request)
        if self._close_pending and not any(getattr(task, "finish_before_close", False) and task.is_alive() for task in self.tasks):
            self.close()

    def _save_path(self, title, suffix, filters):
        return QtWidgets.QFileDialog.getSaveFileName(self, title, str(Path.cwd() / suffix), filters)[0]

    def save_snapshot(self):
        if self.last_pair is None:
            self._error("No measurement to save yet.")
            return
        path = self._save_path("Save snapshot bundle", "alignment.shift.zip", "Snapshot bundle (*.zip)")
        if path:
            pair = self.last_pair
            self._task(lambda: export_bundle(path, pair), lambda _: self.statusBar().showMessage(f"Saved {path}", 10000), finish_before_close=True)

    def export_inspection(self):
        snapshot = self.last_diagnostic
        if snapshot is None:
            self._error("Capture or enable an inspection before exporting it.")
            return
        path = self._save_path("Export inspected images, plots and settings", "inspection.shift.zip", "Inspection bundle (*.zip)")
        if not path:
            return
        buffer = QtCore.QBuffer()
        buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
        self.dock.widget().grab().save(buffer, "PNG")
        plot_png = bytes(buffer.data())
        buffer.close()
        self._task(lambda: export_bundle(path, snapshot.pair, snapshot, plot_png), lambda _: self.statusBar().showMessage(f"Saved {path}", 10000), finish_before_close=True)

    def save_csv(self):
        path = self._save_path("Export measurement history", "measurements.csv", "CSV (*.csv)")
        if path:
            rows = list(self.history)
            self._task(lambda: export_csv(path, rows), lambda _: self.statusBar().showMessage(f"Saved {len(rows)} measurements", 10000), finish_before_close=True)

    def _connect_settings_autosave(self):
        for control in (self.source_kind, self.device, self.view_mode):
            control.currentIndexChanged.connect(self._schedule_settings_save)
        self.tolerance.valueChanged.connect(self._schedule_settings_save)
        self.trend_toggle.toggled.connect(self._schedule_settings_save)
        self.crosshair_toggle.toggled.connect(self._schedule_settings_save)
        self.motion_dock.visibilityChanged.connect(self._schedule_settings_save)
        self.dock.changed.connect(self._schedule_settings_save)
        for control in self.camera_controls.values():
            signal = control.currentIndexChanged if isinstance(control, QtWidgets.QComboBox) else control.valueChanged
            signal.connect(self._schedule_settings_save)

    def _schedule_settings_save(self, *_):
        if not self._restoring_settings and not self._closing:
            self.settings_timer.start()

    def _settings_camera(self):
        self._remembered_camera.update(self._profile_camera)
        self._remembered_camera.update(self._camera_values())
        return dict(self._remembered_camera)

    def _save_startup_settings(self):
        self.settings_timer.stop()
        try:
            self._settings_path.parent.mkdir(parents=True, exist_ok=True)
            ui = dict(source_kind=self.source_kind.currentText(), device_id=self.device.currentData() or "",
                      device_label=self.device.currentText(), view_mode=self.view_mode.currentText(),
                      trend=self.trend_toggle.isChecked(), crosshair=self.crosshair_toggle.isChecked(),
                      motion_visible=not self.motion_dock.isHidden(),
                      analytics_visible=not self.dock.isHidden(), analytics_mode=self.dock.mode.currentText(),
                      analytics_tab=self.dock.tabs.currentIndex(),
                      analytics_pages=[page.enabled.isChecked() for page in self.dock.pages.values()])
            save_profile(self._settings_path, self.config, self.tolerance.value(), self.dock.rate.value(),
                         self._settings_camera(), motion=self.motion_dock.config.to_dict(), ui=ui)
        except Exception as exc:
            self.statusBar().showMessage(f"Could not save startup settings: {exc}", 10000)

    def _apply_settings_profile(self, profile):
        config, tolerance, hz, camera, motion = profile
        motion_config = MotionConfig.from_dict(motion)
        if not isinstance(camera, dict):
            raise ValueError("Invalid saved camera settings")
        if self.reference:
            config.region.bounds(self.reference.frame.shape)
        self._set_config(config)
        self._sync_matching_controls()
        self.tolerance.setValue(tolerance)
        self.dock.rate.setValue(hz)
        self._profile_camera = dict(camera)
        self._remembered_camera = dict(camera)
        self._connect_motion(False)
        self.motion_dock.set_config(motion_config)
        self._motion_config_changed(motion_config)
        if self.capture_worker:
            self._install_controls(self.capture_worker)

    def _restore_startup_settings(self):
        if not self._settings_path.exists():
            return
        try:
            *profile, ui = load_profile(self._settings_path, include_motion=True, include_ui=True)
            self._apply_settings_profile(profile)
            for control, key in ((self.source_kind, "source_kind"), (self.view_mode, "view_mode"),
                                 (self.dock.mode, "analytics_mode")):
                value = ui.get(key)
                if isinstance(value, str) and control.findText(value) >= 0:
                    control.setCurrentText(value)
            identifier = ui.get("device_id", "")
            if isinstance(identifier, str) and identifier:
                self.device.clear()
                label = ui.get("device_label", identifier)
                self.device.addItem(label if isinstance(label, str) else identifier, identifier)
            tab = ui.get("analytics_tab")
            if type(tab) is int and 0 <= tab < self.dock.tabs.count():
                self.dock.tabs.setCurrentIndex(tab)
            for control, key in ((self.trend_toggle, "trend"), (self.crosshair_toggle, "crosshair"),
                                 (self.motion_dock, "motion_visible"),
                                 (self.dock, "analytics_visible")):
                value = ui.get(key)
                if type(value) is bool:
                    if isinstance(control, (QtWidgets.QCheckBox, QtGui.QAction)):
                        control.setChecked(value)
                    else:
                        control.setVisible(value)
            pages = ui.get("analytics_pages", [])
            if isinstance(pages, list):
                for page, enabled in zip(self.dock.pages.values(), pages):
                    if type(enabled) is bool:
                        page.enabled.setChecked(enabled)
            self.notice.setText("Previous settings restored. Connecting hardware automatically.")
        except Exception as exc:
            self.notice.setText(f"Could not restore previous settings: {exc}")

    def save_settings(self):
        path = self._save_path("Save settings profile", "alignment-settings.json", "Settings (*.json)")
        if path:
            camera = self._settings_camera()
            try:
                save_profile(path, self.config, self.tolerance.value(), self.dock.rate.value(), camera,
                             motion=self.motion_dock.config.to_dict())
                self.statusBar().showMessage(f"Saved {path}", 10000)
            except Exception as exc:
                self._error(exc)

    def load_settings(self):
        path = self._choose_file("Load settings profile", "Settings (*.json)")
        if not path:
            return
        try:
            self._apply_settings_profile(load_profile(path, include_motion=True))
            self._schedule_settings_save()
            self.notice.setText("Settings loaded. Apply camera settings as needed; toggle Connect hardware on to enable axis control.")
        except Exception as exc:
            self._error(exc)

    def _fit_images(self):
        for pane in (self.reference_pane, self.live_pane):
            if pane.array is not None:
                h, w = pane.array.shape[:2]
                pane.view.setRange(rect=QtCore.QRectF(0, 0, w, h), padding=.03)

    def _link_views(self, source, target):
        # Empty panes have no image coordinates to share with another view.
        if source.array is None or target.array is None:
            return
        x_range, y_range = source.view.viewRange()
        target.view.setRange(xRange=x_range, yRange=y_range, padding=0)

    def _about(self):
        QtWidgets.QMessageBox.information(self, "Measure Shift", "Connect a camera or start Demo. Capture or load a reference.\n\nWhole-image phase correlation is the default. Draw a rectangle or circle to focus on a feature. X/Y are reference-to-current image displacement; the arrow shows the opposite image correction.\n\nOpen Image analytics for optional live or frozen inspection. Scores are diagnostics, not probabilities.\n\nAcquisition continues while the view is paused. CSV history retains the latest 10,000 measurements.")

    def closeEvent(self, event):
        self._save_startup_settings()
        self.axis_keyboard.clear()
        self.motion.stop_worker()
        if any(getattr(task, "finish_before_close", False) and task.is_alive() for task in self.tasks):
            self._close_pending = True
            self.notice.setText("Finishing the requested export before closing…")
            event.ignore()
            return
        self._closing = True
        QtWidgets.QApplication.instance().removeEventFilter(self.axis_keyboard)
        self.timer.stop()
        self.motion.join(timeout=3)
        if self.motion.is_alive():
            self._closing = False
            self._close_pending = True
            self.notice.setText("Waiting for the linear axis to stop before closing…")
            self.timer.start(33)
            event.ignore()
            return
        if self.capture_worker:
            self.capture_worker.stop()
            self.retired_captures.append(self.capture_worker)
        for worker in (self.measurement, self.analytics, self.preview):
            worker.stop()
        for worker in [*self.retired_captures, self.measurement, self.analytics, self.preview]:
            worker.join(timeout=1)
        event.accept()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Live camera alignment with optional image analytics")
    parser.add_argument("--demo", action="store_true", help="Start the synthetic moving target")
    parser.add_argument("--source", choices=["demo", "ids", "basler", "images", "video"], help="Override the remembered source")
    parser.add_argument("--path", default="", help="Current image, sequence folder, or video path")
    parser.add_argument("--device", default="", help="Camera identifier returned by discovery")
    parser.add_argument("--reference", default="", help="Load a reference image on startup")
    args = parser.parse_args(argv)
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    application.setApplicationName("Measure Shift")
    application.setOrganizationName("Measure Shift")
    window = MainWindow()
    window.show()
    def start():
        kind = "demo" if args.demo else args.source or window.source_kind.currentText().lower()
        remembered = (window.device.currentData() or "") if kind == window.source_kind.currentText().lower() else ""
        identifier = (args.device or remembered) if kind in ("ids", "basler") else ""
        path = (args.path or remembered) if kind in ("images", "video") else args.path
        window.start_source(kind, identifier=identifier, path=path,
                            auto_reference=kind == "demo" and not args.reference)
        if args.reference:
            window._task(lambda: FramePacket.from_array(read_gray(args.reference), Path(args.reference).name), window.set_reference)
    QtCore.QTimer.singleShot(0, start)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
