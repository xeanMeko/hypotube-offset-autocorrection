"""Qt controls for the single linear axis; hardware stays in the motion worker."""
from __future__ import annotations

from dataclasses import replace

from PySide6 import QtWidgets
from PySide6 import QtCore

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

    def __init__(self, toolbar):
        super().__init__(toolbar)
        self.connect_button = QtWidgets.QPushButton("Connect hardware")
        self.connect_button.setObjectName("axisConnectButton")
        self.connect_button.setCheckable(True)
        self.connect_button.setToolTip("Connect the joystick interface and enable movement. Release held directions before moving. Toggle off to stop and disconnect.")
        self.connection_revision = 0
        self.usb_setup_pending = False
        self.calibrate_button = QtWidgets.QPushButton("Run pixel calibration")
        self.calibrate_button.setObjectName("pixelCalibrationButton")
        self.align_button = QtWidgets.QPushButton("Align X")
        self.align_button.setObjectName("alignXButton")
        self.align_button.setToolTip(
            "Use a fresh camera measurement after each relay nudge to align X to the reference.")
        self.stop_button = QtWidgets.QPushButton("STOP")
        self.stop_button.setObjectName("axisStopButton")
        for button in (self.connect_button, self.calibrate_button, self.align_button, self.stop_button):
            toolbar.addWidget(button)
        self.usb_access_button = QtWidgets.QPushButton("Enable Master Brick access")
        self.usb_access_button.setToolTip("Ask an administrator to enable USB access on this PC and retry the connection.")
        self.usb_access_action = toolbar.addWidget(self.usb_access_button)
        self.usb_access_action.setVisible(False)
        self.status = QtWidgets.QLabel("Disconnected · motion disabled")
        self.status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.status.setWordWrap(False)
        self.status.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.status.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self.set_calibration_move(MotionConfig.calibration_presses)
        for button in (self.calibrate_button, self.align_button, self.stop_button):
            button.setEnabled(False)

    def set_calibration_move(self, presses):
        direction = "right" if presses > 0 else "left"
        unit = "press" if abs(presses) == 1 else "presses"
        self.calibrate_button.setToolTip(
            f"Capture a fresh live image, send {abs(presses):,} {direction} "
            f"relay {unit}, and measure the pixel movement. "
            f"Keep a textured feature visible. Change the test move in {MOTION_SETUP_TITLE} → Pixel calibration.")

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
            self.connect_button.setText("Disconnect hardware" if status.connected else "Connect hardware")
            self.connect_button.setEnabled(not self.usb_setup_pending)
        self.usb_access_action.setVisible(status.usb_access_needed and status.connection_revision == self.connection_revision)
        self.stop_button.setEnabled(status.connected)
        self.stop_button.setToolTip("Release the relays and cancel further presses. The axis controller handles any nudge already accepted.")
        if self.usb_setup_pending:
            self.status.setText("Waiting for administrator approval for Master Brick USB access")
        else:
            self.status.setText(f"{status.phase.upper()} · {status.message}")
        self.status.setToolTip(self.status.text())
        idle = not status.busy and status.phase in ("idle", "ready", "aligned", "stopped")
        self.calibrate_button.setEnabled(status.armed and idle and context.live_camera_ready)
        self.align_button.setEnabled(status.armed and idle and context.camera_ready)


class MotionDock(QtWidgets.QDockWidget):
    config_changed = QtCore.Signal(object)
    arm_requested = QtCore.Signal()
    known_distance_requested = QtCore.Signal(float)
    error = QtCore.Signal(str)

    def __init__(self, config=None, parent=None):
        super().__init__(MOTION_SETUP_TITLE, parent)
        self.setObjectName("linearAxisDock")
        self._config = config or MotionConfig()
        self._syncing = False
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        connection = QtWidgets.QGroupBox("Bricklet connection")
        form = QtWidgets.QFormLayout(connection)
        self.host = QtWidgets.QLineEdit()
        self.port = QtWidgets.QSpinBox()
        self.port.setRange(1, 65535)
        self.digital_uid = QtWidgets.QLineEdit()
        self.output_uid = QtWidgets.QLineEdit()
        self.digital_uid.setToolTip("Leave empty for relay-only operation without digital input triggers.")
        for label, control in (("Host", self.host), ("Port", self.port),
                               ("Digital In UID", self.digital_uid),
                               ("Dual Relay UID", self.output_uid)):
            form.addRow(label, control)
        self.pulse_ms = QtWidgets.QSpinBox()
        self.pulse_ms.setRange(1, 5000)
        self.pulse_ms.setToolTip("Time the selected relay closes its A contact for one joystick press.")
        self.release_ms = QtWidgets.QSpinBox()
        self.release_ms.setRange(1, 10000)
        self.release_ms.setToolTip("Minimum time after release before the next press. Allow the axis to finish its nudge.")
        for label, field in (("Joystick press", self.pulse_ms), ("Release interval", self.release_ms)):
            field.setSuffix(" ms")
            field.setKeyboardTracking(False)
            form.addRow(label, field)
        form.addRow(QtWidgets.QLabel("Relay A0 → IN 1     ·     Relay A1 → IN 2"))
        layout.addWidget(connection)

        inputs = QtWidgets.QGroupBox("Digital inputs")
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
        self.reference_channel.setToolTip("Capture the current reference on LOW → HIGH, at most once per second. Independent of joystick polarity.")
        self.align_channel.setToolTip("Start direct X alignment on LOW → HIGH. Requires a live camera, a reference, and enabled movement.")
        self.active_high = QtWidgets.QCheckBox("Joystick active high")
        self.apply_inputs_button = QtWidgets.QPushButton("Apply digital inputs")
        self.apply_inputs_button.setToolTip("Apply all channel assignments and joystick polarity together.")
        self.inputs = QtWidgets.QLabel("0: —   1: —   2: —   3: —")
        form.addRow("Left", self.left_channel)
        form.addRow("Right", self.right_channel)
        form.addRow("Reference capture", self.reference_channel)
        form.addRow("Align X", self.align_channel)
        form.addRow(self.active_high)
        form.addRow(self.apply_inputs_button)
        hint = QtWidgets.QLabel("Reference and Align X trigger on LOW → HIGH. Release before pressing again.")
        hint.setWordWrap(True)
        form.addRow(hint)
        form.addRow("Raw inputs", self.inputs)
        layout.addWidget(inputs)

        jogging = QtWidgets.QGroupBox("Manual movement")
        form = QtWidgets.QFormLayout(jogging)
        self.arm_button = QtWidgets.QPushButton("Enable joystick")
        self.arm_button.setToolTip("Enable app movement with the joystick and arrow keys released.")
        self.frequency = self._number(.1, 10, 1)
        self.frequency.setSuffix(" Hz")
        self.frequency.setSingleStep(.1)
        form.addRow("Nudge repeat rate", self.frequency)
        form.addRow(self.arm_button)
        layout.addWidget(jogging)

        calibration = QtWidgets.QGroupBox("Pixel calibration")
        form = QtWidgets.QFormLayout(calibration)
        self.pixel_scale = QtWidgets.QLabel("Pixels: not calibrated")
        self.pixel_scale.setWordWrap(True)
        self.test_presses = QtWidgets.QSpinBox()
        self.test_presses.setRange(-1000000, 1000000)
        self.test_presses.setKeyboardTracking(False)
        self.test_presses.setToolTip("Number of joystick presses for calibration. Positive sends right relay pulses and negative sends left relay pulses.")
        self.known_pixel_distance = self._number(-1e9, 1e9, 8)
        self.known_pixel_distance.setSuffix(" px")
        self.known_pixel_distance.setToolTip("X movement for one right joystick press. Enter negative pixels if the image moves left.")
        self.use_known_distance_button = QtWidgets.QPushButton("Use known distance")
        self.use_known_distance_button.setToolTip("Save a measured displacement per right press for the live camera.")
        form.addRow(self.pixel_scale)
        form.addRow("Known X / right press", self.known_pixel_distance)
        form.addRow(self.use_known_distance_button)
        form.addRow("Test presses (+ right / − left)", self.test_presses)
        layout.addWidget(calibration)
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
        self.set_config(self._config)
        self.set_connected(False)

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
            self.pixel_scale.setText(f"Pixels: {scale:+.6g} px / right press" if scale is not None else "Pixels: not calibrated")
            self.known_pixel_distance.setValue(scale or 0.)
        finally:
            self._syncing = False
        self._inputs_edited()

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
        self.use_known_distance_button.setEnabled(False)

    def display_status(self, status, context):
        # Set each action's final state once. Temporarily disabling a button
        # during a refresh cancels a mouse press before its release arrives.
        self._display_connection(status.connected)
        self.inputs.setText("   ".join(f"{i}: {'HIGH' if value else 'low'}" for i, value in enumerate(status.inputs4))
                            if self._config.digital_uid else "Digital In monitor disabled")
        idle = not status.busy and status.phase in ("idle", "ready", "aligned", "stopped")
        self.arm_button.setEnabled(status.connected and not status.armed)
        live = context.live_camera_ready
        self.use_known_distance_button.setEnabled(
            live and (idle or status.phase in ("disconnected", "connected"))
            and not status.busy)
