"""Qt controls for the single linear axis; hardware stays in the motion worker."""
from __future__ import annotations

from dataclasses import replace

from PySide6 import QtWidgets
from PySide6 import QtCore, QtGui

from .i18n import Language, Term
from .joystick import INPUT_DEBOUNCE_S
from .motion import MotionConfig


MOTION_SETUP_TITLE = "Motion Setup"


class AxisKeyboardFilter(QtCore.QObject):
    """Merge physical key state rather than relying on operating-system repeat."""

    changed = QtCore.Signal(bool, bool)

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.left = self.right = False

    @staticmethod
    def editing(widget):
        excluded = (QtWidgets.QLineEdit, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit,
                    QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox, QtWidgets.QAbstractItemView,
                    QtWidgets.QAbstractSlider, QtWidgets.QMenu, QtWidgets.QMenuBar, QtWidgets.QTabBar)
        while widget is not None:
            if isinstance(widget, excluded):
                return True
            widget = widget.parentWidget()
        return False

    def clear(self):
        if self.left or self.right:
            self.left = self.right = False
            self.changed.emit(False, False)

    def eventFilter(self, watched, event):
        kind = event.type()
        if kind == QtCore.QEvent.Type.ApplicationDeactivate or (
                kind == QtCore.QEvent.Type.WindowDeactivate and watched is self.window):
            self.clear()
        if kind == QtCore.QEvent.Type.FocusIn and isinstance(watched, QtWidgets.QWidget) and self.editing(watched):
            self.clear()
        if kind == QtCore.QEvent.Type.Show and isinstance(watched, (QtWidgets.QMenu, QtWidgets.QDialog)):
            self.clear()
        if kind not in (QtCore.QEvent.Type.KeyPress, QtCore.QEvent.Type.KeyRelease,
                        QtCore.QEvent.Type.ShortcutOverride):
            return False
        if event.key() not in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Right):
            return False
        app = QtWidgets.QApplication.instance()
        eligible = (app.activeWindow() is self.window and app.activeModalWidget() is None
                    and app.activePopupWidget() is None and not self.editing(app.focusWidget())
                    and not (isinstance(watched, QtWidgets.QWidget) and self.editing(watched)))
        if not eligible:
            self.clear()
            return False
        if event.modifiers() & (QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.AltModifier
                                | QtCore.Qt.KeyboardModifier.MetaModifier | QtCore.Qt.KeyboardModifier.ShiftModifier):
            self.clear()
            return False
        if kind == QtCore.QEvent.Type.ShortcutOverride:
            event.accept()
            return True
        if event.isAutoRepeat():
            return True
        pressed = kind == QtCore.QEvent.Type.KeyPress
        previous = self.left, self.right
        if event.key() == QtCore.Qt.Key.Key_Left:
            self.left = pressed
        else:
            self.right = pressed
        if previous != (self.left, self.right):
            self.changed.emit(self.left, self.right)
        event.accept()
        return True


class MotionControls(QtCore.QObject):
    """Everyday axis actions and status readout."""

    def __init__(self, toolbar, language=None):
        super().__init__(toolbar)
        self.ui = language or Language(parent=self)
        self.compact = self.connected = False
        self.connect_button = self.ui.button("Connect hardware")
        self.connect_button.setObjectName("axisConnectButton")
        self.connect_button.setCheckable(True)
        self.ui.bind(self.connect_button.setToolTip, "Connect the joystick interface and enable movement. Release held directions before moving. Toggle off to stop and disconnect.")
        self.connection_revision = 0
        self.usb_setup_pending = False
        self.align_button = self.ui.button("Align X")
        self.align_button.setObjectName("alignXButton")
        self.ui.bind(self.align_button.setToolTip,
            "Use a fresh camera measurement after each relay nudge to align X to the reference.")
        self.stop_button = self.ui.button("STOP")
        self.stop_button.setObjectName("axisStopButton")
        for button in (self.connect_button, self.align_button, self.stop_button):
            toolbar.addWidget(button)
        self.usb_access_button = self.ui.button("Enable Master Brick access")
        self.ui.bind(self.usb_access_button.setToolTip, "Ask an administrator to enable USB access on this PC and retry the connection.")
        self.usb_access_action = toolbar.addWidget(self.usb_access_button)
        self.usb_access_action.setVisible(False)
        self.status = self.ui.label("Disconnected · motion disabled")
        self.status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.status.setWordWrap(False)
        self.status.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.status.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        for button in (self.align_button, self.stop_button):
            button.setEnabled(False)

    def _connection_caption(self, connected):
        if self.compact:
            return "Disconnect" if connected else "Connect"
        return "Disconnect hardware" if connected else "Connect hardware"

    def set_compact(self, compact):
        self.compact = compact
        self.ui.bind(self.connect_button.setText, self._connection_caption(self.connected))
        button = self.connect_button
        button.ensurePolished()
        metrics = button.fontMetrics()
        padding = button.sizeHint().width() - metrics.horizontalAdvance(button.text())
        button.setFixedWidth(max(metrics.horizontalAdvance(self.ui.text(self._connection_caption(state)))
                                 for state in (False, True)) + padding + 2)

    def set_usb_setup_pending(self, pending):
        self.usb_setup_pending = pending
        self.usb_access_button.setEnabled(not pending)
        if pending:
            self.connect_button.setEnabled(False)

    def set_connection_pending(self, connected, revision):
        self.connection_revision = revision
        with QtCore.QSignalBlocker(self.connect_button):
            self.connect_button.setChecked(connected)
        self.connect_button.setEnabled(False)

    def display_status(self, status, context):
        # Set each action's final state once so refreshes do not cancel a mouse press.
        if status.connection_revision >= self.connection_revision:
            with QtCore.QSignalBlocker(self.connect_button):
                self.connect_button.setChecked(status.connected)
            self.connected = status.connected
            self.ui.bind(self.connect_button.setText, self._connection_caption(status.connected))
            self.connect_button.setEnabled(not self.usb_setup_pending)
        self.usb_access_action.setVisible(status.usb_access_needed and status.connection_revision == self.connection_revision)
        self.stop_button.setEnabled(status.connected)
        self.ui.bind(self.stop_button.setToolTip, "Release the relays and cancel further presses. The axis controller handles any nudge already accepted.")
        if self.usb_setup_pending:
            message = ("Waiting for administrator approval for Master Brick USB access",)
        else:
            message = ("{} · {}", Term(status.phase.upper()), Term(status.message))
        self.ui.bind(self.status.setText, *message)
        self.ui.bind(self.status.setToolTip, *message)
        idle = not status.busy and status.phase in ("idle", "ready", "aligned", "stopped")
        self.align_button.setEnabled(status.armed and idle and context.camera_ready)


class PulsePreview(QtWidgets.QWidget):
    """One ideal relay cycle for a held direction, drawn to a common time scale."""

    def __init__(self, language, config):
        super().__init__()
        self.ui = language
        self.config = config
        self.ui.bind(self.setAccessibleName, "Relay timing preview")
        self.ui.changed.connect(self.update)

    def sizeHint(self):
        return QtCore.QSize(200, self.fontMetrics().height() * 4)

    def set_config(self, config):
        self.config = config
        self.update()

    @property
    def release_ms(self):
        floor = INPUT_DEBOUNCE_S * 1000 if self.config.digital_uid else 0
        return max(self.config.release_ms, floor)

    @property
    def period_ms(self):
        return max(self.config.pulse_ms + self.release_ms, 1000 / self.config.repeat_hz)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        metrics = self.fontMetrics()
        closed, opened = self.ui.text("Contact closed"), self.ui.text("Contact open")
        left = max(metrics.horizontalAdvance(label) for label in (closed, opened)) + 10
        right = self.width() - 5
        high, low = metrics.height(), 2 * metrics.height()
        text_color = self.palette().windowText().color()
        painter.setPen(text_color)
        painter.drawText(0, high + metrics.ascent() // 3, closed)
        painter.drawText(0, low + metrics.ascent() // 3, opened)
        painter.drawText(left, low + metrics.height() + 2, "0 ms")
        end_label = self.ui.text("Earliest repeat · {:g} ms", round(self.period_ms, 1))
        if metrics.horizontalAdvance(end_label) + metrics.horizontalAdvance("0 ms") + 8 > right - left:
            end_label = f"{round(self.period_ms, 1):g} ms"
        painter.drawText(right - metrics.horizontalAdvance(end_label), low + metrics.height() + 2, end_label)
        scale = (right - left) / self.period_ms
        fall = left + self.config.pulse_ms * scale
        ready = fall + self.release_ms * scale
        pen = QtGui.QPen(text_color, 2)
        painter.setPen(pen)
        painter.drawPolyline([QtCore.QPointF(x, y) for x, y in (
            (left, low), (left, high), (fall, high), (fall, low), (ready, low))])
        pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(QtCore.QLineF(ready, low, right, low))
        pen.setStyle(QtCore.Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.drawLine(right, low, right, high)


class MotionDock(QtWidgets.QDockWidget):
    config_changed = QtCore.Signal(object)
    arm_requested = QtCore.Signal()
    known_distance_requested = QtCore.Signal(float)
    error = QtCore.Signal(str)

    def __init__(self, config=None, parent=None, language=None):
        super().__init__(parent)
        self.ui = language or Language(parent=self)
        self.ui.bind(self.setWindowTitle, MOTION_SETUP_TITLE)
        self.setObjectName("linearAxisDock")
        self._config = config or MotionConfig()
        self._syncing = False
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        timing = self.ui.group("Relay pulse timing")
        form = QtWidgets.QFormLayout(timing)
        self.pulse_ms = QtWidgets.QSpinBox()
        self.pulse_ms.setRange(1, 5000)
        self.ui.bind(self.pulse_ms.setToolTip,
                     "Time the relay contact stays closed. Bench Align X must detect Digital In during this pulse.")
        self.release_ms = QtWidgets.QSpinBox()
        self.release_ms.setRange(1, 10000)
        self.ui.bind(self.release_ms.setToolTip, "Minimum time after release before the next press. Allow the axis to finish its nudge.")
        for label, field in (("Pulse duration", self.pulse_ms), ("Release interval", self.release_ms)):
            field.setSuffix(" ms")
            field.setKeyboardTracking(False)
            form.addRow(self.ui.label(label), field)
        self.frequency = self._number(.1, 10, 1)
        self.frequency.setSuffix(" Hz")
        self.frequency.setSingleStep(.1)
        self.ui.bind(self.frequency.setToolTip,
                     "Maximum pulses per second. Motor travel and camera feedback can delay the next nudge.")
        form.addRow(self.ui.label("Nudge repeat rate"), self.frequency)
        self.preview = PulsePreview(self.ui, self._config)
        form.addRow(self.preview)
        self.timing_summary = QtWidgets.QLabel()
        self.timing_summary.setWordWrap(True)
        form.addRow(self.timing_summary)
        self.release_hint = QtWidgets.QLabel()
        self.ui.bind(self.release_hint.setText, "Digital In allows at least {} ms for contact settling.",
                     round(INPUT_DEBOUNCE_S * 1000))
        self.release_hint.setWordWrap(True)
        form.addRow(self.release_hint)
        layout.addWidget(timing)

        connection = self.ui.group("Bricklet connection")
        form = QtWidgets.QFormLayout(connection)
        self.host = QtWidgets.QLineEdit()
        self.port = QtWidgets.QSpinBox()
        self.port.setRange(1, 65535)
        self.digital_uid = QtWidgets.QLineEdit()
        self.output_uid = QtWidgets.QLineEdit()
        self.ui.bind(self.digital_uid.setToolTip, "Leave empty for relay-only operation without digital input triggers.")
        for label, control in (("Host", self.host), ("Port", self.port),
                               ("Digital In UID", self.digital_uid),
                               ("Dual Relay UID", self.output_uid)):
            form.addRow(self.ui.label(label), control)
        form.addRow(self.ui.label("Relay A0 → IN 1     ·     Relay A1 → IN 2"))
        layout.addWidget(connection)

        inputs = self.ui.group("Digital inputs")
        form = QtWidgets.QFormLayout(inputs)
        self.left_channel = QtWidgets.QComboBox()
        self.right_channel = QtWidgets.QComboBox()
        for control in (self.left_channel, self.right_channel):
            for channel in range(4):
                control.addItem(f"Channel {channel}", channel)
        self.reference_channel = QtWidgets.QComboBox()
        self.align_channel = QtWidgets.QComboBox()
        for control in (self.reference_channel, self.align_channel):
            control.addItem("Disabled", None)
            for channel in range(4):
                control.addItem(f"Channel {channel}", channel)
        self.ui.bind(self.reference_channel.setToolTip, "Capture the current reference on LOW → HIGH, at most once per second. Independent of joystick polarity.")
        self.ui.bind(self.align_channel.setToolTip, "Start direct X alignment on LOW → HIGH. Requires a live camera, a reference, and enabled movement.")
        self.active_high = self.ui.check("Joystick active high")
        self.apply_inputs_button = self.ui.button("Apply digital inputs")
        self.ui.bind(self.apply_inputs_button.setToolTip, "Apply all channel assignments and joystick polarity together.")
        self.inputs = self.ui.label("0: —   1: —   2: —   3: —")
        form.addRow(self.ui.label("Left"), self.left_channel)
        form.addRow(self.ui.label("Right"), self.right_channel)
        form.addRow(self.ui.label("Reference capture"), self.reference_channel)
        form.addRow(self.ui.label("Align X"), self.align_channel)
        form.addRow(self.active_high)
        form.addRow(self.apply_inputs_button)
        hint = self.ui.label("Reference and Align X trigger on LOW → HIGH. Release before pressing again.")
        hint.setWordWrap(True)
        form.addRow(hint)
        form.addRow(self.ui.label("Raw inputs"), self.inputs)
        layout.addWidget(inputs)

        jogging = self.ui.group("Manual movement")
        form = QtWidgets.QFormLayout(jogging)
        self.arm_button = self.ui.button("Enable joystick")
        self.ui.bind(self.arm_button.setToolTip, "Enable app movement with the joystick and arrow keys released.")
        form.addRow(self.arm_button)
        layout.addWidget(jogging)

        calibration = self.ui.group("Pixel calibration")
        form = QtWidgets.QFormLayout(calibration)
        self.pixel_scale = self.ui.label("Pixels: not calibrated")
        self.pixel_scale.setWordWrap(True)
        self.test_presses = QtWidgets.QSpinBox()
        self.test_presses.setRange(-1000000, 1000000)
        self.test_presses.setKeyboardTracking(False)
        self.ui.bind(self.test_presses.setToolTip, "Number of joystick presses for calibration. Positive sends right relay pulses and negative sends left relay pulses.")
        self.known_pixel_distance = self._number(-1e9, 1e9, 8)
        self.known_pixel_distance.setSuffix(" px")
        self.ui.bind(self.known_pixel_distance.setToolTip, "X movement for one right joystick press. Enter negative pixels if the image moves left.")
        self.use_known_distance_button = self.ui.button("Use known distance")
        self.ui.bind(self.use_known_distance_button.setToolTip, "Save a measured displacement per right press for the live camera.")
        self.calibrate_button = self.ui.button("Run pixel calibration")
        self.calibrate_button.setObjectName("pixelCalibrationButton")
        form.addRow(self.pixel_scale)
        form.addRow(self.ui.label("Known X / right press"), self.known_pixel_distance)
        form.addRow(self.use_known_distance_button)
        form.addRow(self.ui.label("Test presses (+ right / − left)"), self.test_presses)
        form.addRow(self.calibrate_button)
        layout.addWidget(calibration)
        for group in (timing, connection, inputs, calibration):
            group.layout().setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        layout.addStretch()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        self.setWidget(scroll)
        self.setMinimumWidth(320)
        for control in (self.host, self.digital_uid, self.output_uid):
            control.editingFinished.connect(self._edited)
        for control in (self.port, self.frequency, self.test_presses, self.pulse_ms, self.release_ms):
            control.valueChanged.connect(self._edited)
        for control in (self.left_channel, self.right_channel, self.reference_channel, self.align_channel):
            control.currentIndexChanged.connect(self._inputs_edited)
        self.active_high.toggled.connect(self._inputs_edited)
        self.apply_inputs_button.clicked.connect(self._apply_inputs)
        self.arm_button.clicked.connect(self.arm_requested)
        self.use_known_distance_button.clicked.connect(
            lambda: self.known_distance_requested.emit(self.known_pixel_distance.value()))
        self.ui.changed.connect(self._translate_channels)
        self._translate_channels()
        self.set_config(self._config)
        self.test_presses.valueChanged.connect(self._update_calibration_tooltip)
        self._update_calibration_tooltip()
        self.set_connected(False)

    def _update_calibration_tooltip(self):
        presses = self.test_presses.value()
        direction = Term("right" if presses > 0 else "left")
        unit = Term("press" if abs(presses) == 1 else "presses")
        self.ui.bind(self.calibrate_button.setToolTip,
                     "Capture a fresh live image, send {0:,} {1} relay {2}, and measure the pixel movement. "
                     "Keep a textured feature visible.",
                     abs(presses), direction, unit)

    def show_pulse_settings(self):
        self.show()
        self.raise_()
        self.widget().verticalScrollBar().setValue(0)
        self.pulse_ms.setFocus()

    def _refresh_preview(self):
        self.preview.set_config(self._config)
        release = self.preview.release_ms
        self.release_hint.setVisible(release > self._config.release_ms)
        extra = max(0, self.preview.period_ms - self._config.pulse_ms - release)
        self.ui.bind(self.timing_summary.setText,
                     "Closed for {:g} ms, then open for at least {:g} ms. "
                     "The repeat rate adds {:g} ms of wait (dashed).",
                     self._config.pulse_ms, release, round(extra, 1))
        self.ui.bind(self.preview.setAccessibleDescription,
                     "Earliest repeat after {:g} ms.", round(self.preview.period_ms, 1))

    def _translate_channels(self):
        for control in (self.left_channel, self.right_channel, self.reference_channel, self.align_channel):
            with QtCore.QSignalBlocker(control):
                for index in range(control.count()):
                    channel = control.itemData(index)
                    text = self.ui.text("Disabled") if channel is None else self.ui.text("Channel {}", channel)
                    control.setItemText(index, text)

    @staticmethod
    def _number(low, high, decimals):
        control = QtWidgets.QDoubleSpinBox()
        control.setDecimals(decimals)
        control.setRange(low, high)
        control.setKeyboardTracking(False)
        return control

    @property
    def config(self):
        return self._config

    def _input_values(self):
        return dict(left_channel=self.left_channel.currentData(), right_channel=self.right_channel.currentData(),
                    reference_channel=self.reference_channel.currentData(), align_channel=self.align_channel.currentData(),
                    active_high=self.active_high.isChecked())

    def _inputs_edited(self):
        if not self._syncing:
            self.apply_inputs_button.setEnabled(
                any(value != getattr(self._config, name) for name, value in self._input_values().items()))

    def _apply_inputs(self):
        old = self._config
        try:
            config = replace(old, **self._input_values()).validated()
        except ValueError as exc:
            self.error.emit(str(exc))
            return
        if (config.left_channel, config.right_channel) != (old.left_channel, old.right_channel):
            config = config.without_press_scale()
        self.set_config(config)
        self._inputs_edited()
        if config != old:
            self.config_changed.emit(config)

    def set_config(self, config, *, preserve_input_edits=False):
        input_values = self._input_values()
        if not preserve_input_edits or not self.apply_inputs_button.isEnabled():
            input_values = {name: getattr(config, name) for name in input_values}
        self._syncing = True
        try:
            self._config = config
            self.pulse_ms.setValue(config.pulse_ms)
            self.release_ms.setValue(config.release_ms)
            for control, value in ((self.host, config.host),
                                   (self.digital_uid, config.digital_uid), (self.output_uid, config.output_uid)):
                control.setText(value)
            self.port.setValue(config.port)
            for name, value in input_values.items():
                control = getattr(self, name)
                if isinstance(control, QtWidgets.QComboBox):
                    control.setCurrentIndex(control.findData(value))
                else:
                    control.setChecked(value)
            self.frequency.setValue(config.repeat_hz)
            self.test_presses.setValue(config.calibration_presses)
            scale = config.pixels_per_right_press
            if scale is None:
                self.ui.bind(self.pixel_scale.setText, "Pixels: not calibrated")
            else:
                self.ui.bind(self.pixel_scale.setText, "Pixels: {:+.6g} px / right press", scale)
            self.known_pixel_distance.setValue(scale or 0.)
        finally:
            self._syncing = False
        self._inputs_edited()
        self._refresh_preview()

    def _edited(self):
        if self._syncing:
            return
        old = self._config
        try:
            config = replace(old, host=self.host.text().strip(), port=self.port.value(),
                             output_uid=self.output_uid.text().strip(), digital_uid=self.digital_uid.text().strip(),
                             pulse_ms=self.pulse_ms.value(), release_ms=self.release_ms.value(),
                             repeat_hz=self.frequency.value(), calibration_presses=self.test_presses.value()).validated()
            if any(getattr(config, key) != getattr(old, key) for key in
                   ("host", "port", "output_uid", "pulse_ms")):
                config = config.without_press_scale()
                self.set_config(config, preserve_input_edits=True)
            else:
                self._config = config
                self._refresh_preview()
            if config != old:
                self.config_changed.emit(config)
        except ValueError as exc:
            self.set_config(old, preserve_input_edits=True)
            self.error.emit(str(exc))

    def _display_connection(self, connected):
        for control in (self.host, self.port, self.digital_uid, self.output_uid):
            control.setEnabled(not connected)

    def set_connected(self, connected):
        self._display_connection(connected)
        self.arm_button.setEnabled(False)
        self.calibrate_button.setEnabled(False)
        self.use_known_distance_button.setEnabled(False)

    def display_status(self, status, context):
        # Set each action's final state once. Temporarily disabling a button
        # during a refresh cancels a mouse press before its release arrives.
        self._display_connection(status.connected)
        self.ui.bind(self.inputs.setText, "   ".join(f"{i}: {'HIGH' if value else 'low'}" for i, value in enumerate(status.inputs4))
                            if self._config.digital_uid else "Digital In monitor disabled")
        idle = not status.busy and status.phase in ("idle", "ready", "aligned", "stopped")
        self.arm_button.setEnabled(status.connected and not status.armed)
        live = context.live_camera_ready
        self.calibrate_button.setEnabled(status.armed and idle and live)
        self.use_known_distance_button.setEnabled(
            live and (idle or status.phase in ("disconnected", "connected"))
            and not status.busy)
