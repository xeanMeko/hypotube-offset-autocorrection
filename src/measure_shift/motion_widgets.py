"""Qt controls for the single linear axis; hardware stays in the motion worker."""
from __future__ import annotations

from dataclasses import replace

from PySide6 import QtWidgets
from PySide6 import QtCore

from .motion import MotionConfig


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


class MotionDock(QtWidgets.QDockWidget):
    config_changed = QtCore.Signal(object)
    connect_requested = QtCore.Signal()
    setup_requested = QtCore.Signal()
    arm_requested = QtCore.Signal()
    stop_requested = QtCore.Signal()
    calibrate_requested = QtCore.Signal(int)
    known_distance_requested = QtCore.Signal(float)
    align_requested = QtCore.Signal()
    error = QtCore.Signal(str)

    def __init__(self, config=None, parent=None):
        super().__init__("Linear axis", parent)
        self.setObjectName("linearAxisDock")
        self._config = config or MotionConfig()
        self._syncing = False
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        self.status = QtWidgets.QLabel("Disconnected · motion disabled")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        connection = QtWidgets.QGroupBox("Bricklet connection")
        form = QtWidgets.QFormLayout(connection)
        self.host = QtWidgets.QLineEdit()
        self.port = QtWidgets.QSpinBox()
        self.port.setRange(1, 65535)
        self.stepper_uid = QtWidgets.QLineEdit()
        self.digital_uid = QtWidgets.QLineEdit()
        for label, control in (("Host", self.host), ("Port", self.port),
                               ("Stepper UID", self.stepper_uid), ("Digital In UID", self.digital_uid)):
            form.addRow(label, control)
        self.connect_button = QtWidgets.QPushButton("Connect hardware")
        self.connect_button.setToolTip("Connect, apply motor settings, and enable movement. A held direction must be released before moving.")
        form.addRow(self.connect_button)
        layout.addWidget(connection)

        inputs = QtWidgets.QGroupBox("Joystick wiring")
        form = QtWidgets.QFormLayout(inputs)
        self.left_channel = QtWidgets.QComboBox()
        self.right_channel = QtWidgets.QComboBox()
        for control in (self.left_channel, self.right_channel):
            for channel in range(4):
                control.addItem(f"Channel {channel}", channel)
        self.active_high = QtWidgets.QCheckBox("Active high")
        self.invert_direction = QtWidgets.QCheckBox("Invert motor direction")
        self.inputs = QtWidgets.QLabel("0: —   1: —   2: —   3: —")
        form.addRow("Left / forward", self.left_channel)
        form.addRow("Right / backward", self.right_channel)
        form.addRow(self.active_high)
        form.addRow(self.invert_direction)
        form.addRow("Raw inputs", self.inputs)
        layout.addWidget(inputs)

        reference = QtWidgets.QGroupBox("Digital reference capture")
        form = QtWidgets.QFormLayout(reference)
        self.reference_channel = QtWidgets.QComboBox()
        self.reference_channel.addItem("Disabled", None)
        for channel in range(4):
            self.reference_channel.addItem(f"Channel {channel}", channel)
        self.reference_channel.setToolTip("Capture the current reference on LOW → HIGH, at most once per second. Independent of joystick polarity.")
        form.addRow("Reference capture channel", self.reference_channel)
        hint = QtWidgets.QLabel("LOW → HIGH · maximum 1 capture / second")
        hint.setWordWrap(True)
        form.addRow(hint)
        layout.addWidget(reference)

        motor = QtWidgets.QGroupBox("Motor setup")
        form = QtWidgets.QFormLayout(motor)
        self.device_settings = QtWidgets.QLabel("Connect to read motor settings")
        self.device_settings.setWordWrap(True)
        form.addRow(self.device_settings)
        self.motor_current = QtWidgets.QSpinBox()
        self.motor_current.setRange(0, 1640)
        self.motor_current.setSpecialValueText("Set rated current")
        self.motor_current.setSuffix(" mA")
        self.resolution = QtWidgets.QComboBox()
        for denominator in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            self.resolution.addItem("Full step" if denominator == 1 else f"1/{denominator} step", denominator)
        self.interpolation = QtWidgets.QCheckBox("Interpolate microsteps")
        self.setup_button = QtWidgets.QPushButton("Apply motor setup")
        self.arm_button = QtWidgets.QPushButton("Enable motor")
        self.arm_button.setToolTip("Apply the selected motor settings and enable movement. Leave the joystick and arrow keys released.")
        form.addRow("Motor current", self.motor_current)
        form.addRow("Step resolution", self.resolution)
        form.addRow(self.interpolation)
        self.advanced_motor_toggle = QtWidgets.QToolButton()
        self.advanced_motor_toggle.setText("Advanced motor settings")
        self.advanced_motor_toggle.setCheckable(True)
        self.advanced_motor_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_motor_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.advanced_motor_settings = QtWidgets.QWidget()
        advanced_form = QtWidgets.QFormLayout(self.advanced_motor_settings)
        advanced_form.setContentsMargins(0, 0, 0, 0)
        self.speed = QtWidgets.QSpinBox()
        self.speed.setRange(1, 65535)
        self.speed.setSuffix(" steps/s")
        self.speed.setKeyboardTracking(False)
        self.speed.setToolTip("Speed for nudges, calibration moves, and Align X, in driver steps per second. Default: 200 steps/s.")
        advanced_form.addRow("Motor speed", self.speed)
        self.advanced_motor_settings.hide()
        self.advanced_motor_toggle.toggled.connect(self._toggle_advanced_motor_settings)
        form.addRow(self.advanced_motor_toggle)
        form.addRow(self.advanced_motor_settings)
        form.addRow(self.setup_button)
        form.addRow(self.arm_button)
        layout.addWidget(motor)

        jogging = QtWidgets.QGroupBox("Nudging · joystick or ← / →")
        form = QtWidgets.QFormLayout(jogging)
        self.unit = QtWidgets.QComboBox()
        for label, value in (("Driver steps", "driversteps"), ("Pixels", "pixels")):
            self.unit.addItem(label, value)
        self.distance = self._number(0.00000001, 1e9, 8)
        self.distance.setToolTip("Distance of each manual nudge and maximum distance of each Align X correction.")
        self.frequency = self._number(.1, 10, 1)
        self.frequency.setSuffix(" Hz")
        self.frequency.setSingleStep(.1)
        self.effective = QtWidgets.QLabel()
        self.effective.setWordWrap(True)
        hint = QtWidgets.QLabel("Hold to nudge one move at a time. Release finishes the current nudge; released inputs are discarded. Arrows keep their normal behavior in settings fields.")
        hint.setWordWrap(True)
        form.addRow("Units", self.unit)
        form.addRow("Distance / nudge", self.distance)
        form.addRow("Repeat frequency", self.frequency)
        form.addRow(self.effective)
        form.addRow(hint)
        layout.addWidget(jogging)

        calibration = QtWidgets.QGroupBox("Pixel calibration")
        form = QtWidgets.QFormLayout(calibration)
        self.pixel_scale = QtWidgets.QLabel("Pixels: not calibrated")
        self.pixel_scale.setWordWrap(True)
        self.test_steps = QtWidgets.QSpinBox()
        self.test_steps.setRange(-1000000, 1000000)
        self.test_steps.setValue(100)
        self.calibrate_button = QtWidgets.QPushButton("Run pixel calibration move")
        self.calibrate_button.setToolTip("Capture the latest live image, move this signed number of driver steps, and compare the new image with that baseline. Keep a textured feature visible.")
        self.known_pixel_distance = self._number(-1e9, 1e9, 8)
        self.known_pixel_distance.setSuffix(" px")
        self.known_pixel_distance.setToolTip("Known X movement for a positive driver-step nudge. Use a negative value if positive motor steps move the image left.")
        self.known_distance_hint = QtWidgets.QLabel()
        self.known_distance_hint.setWordWrap(True)
        self.use_known_distance_button = QtWidgets.QPushButton("Use known distance")
        self.use_known_distance_button.setToolTip("Save the known pixel movement as calibration for the live camera, without moving the motor.")
        form.addRow(self.pixel_scale)
        form.addRow("Known distance / nudge", self.known_pixel_distance)
        form.addRow(self.known_distance_hint)
        form.addRow(self.use_known_distance_button)
        form.addRow("Test move (steps)", self.test_steps)
        form.addRow(self.calibrate_button)
        layout.addWidget(calibration)
        self.align_button = QtWidgets.QPushButton("Align X")
        self.stop_button = QtWidgets.QPushButton("STOP")
        self.stop_button.setStyleSheet("font-weight: bold; background: #773e38")
        layout.addStretch()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        shell = QtWidgets.QWidget()
        shell_layout = QtWidgets.QVBoxLayout(shell)
        shell_layout.setContentsMargins(4, 0, 4, 4)
        shell_layout.addWidget(scroll, 1)
        actions = QtWidgets.QHBoxLayout()
        actions.addWidget(self.align_button, 1)
        actions.addWidget(self.stop_button, 1)
        shell_layout.addLayout(actions)
        self.setWidget(shell)
        self.setMinimumWidth(320)
        for control in (self.host, self.stepper_uid, self.digital_uid, self.motor_current):
            control.editingFinished.connect(self._edited)
        for control in (self.port, self.distance, self.speed, self.frequency):
            control.valueChanged.connect(self._edited)
        for control in (self.left_channel, self.right_channel, self.reference_channel, self.resolution):
            control.currentIndexChanged.connect(self._edited)
        for control in (self.active_high, self.invert_direction, self.interpolation):
            control.toggled.connect(self._edited)
        self.unit.currentIndexChanged.connect(self._unit_changed)
        self.connect_button.clicked.connect(self.connect_requested)
        self.setup_button.clicked.connect(self.setup_requested)
        self.arm_button.clicked.connect(self.arm_requested)
        self.stop_button.clicked.connect(self.stop_requested)
        self.calibrate_button.clicked.connect(lambda: self.calibrate_requested.emit(self.test_steps.value()))
        self.use_known_distance_button.clicked.connect(
            lambda: self.known_distance_requested.emit(self.known_pixel_distance.value()))
        self.align_button.clicked.connect(self.align_requested)
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

    def _toggle_advanced_motor_settings(self, expanded):
        self.advanced_motor_settings.setVisible(expanded)
        self.advanced_motor_toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if expanded else QtCore.Qt.ArrowType.RightArrow)

    def set_config(self, config):
        self._syncing = True
        try:
            self._config = config
            for control, value in ((self.host, config.host), (self.stepper_uid, config.stepper_uid),
                                   (self.digital_uid, config.digital_uid)):
                control.setText(value)
            self.port.setValue(config.port)
            self.left_channel.setCurrentIndex(config.left_channel)
            self.right_channel.setCurrentIndex(config.right_channel)
            self.reference_channel.setCurrentIndex(self.reference_channel.findData(config.reference_channel))
            self.active_high.setChecked(config.active_high)
            self.invert_direction.setChecked(config.invert_direction)
            self.motor_current.setValue(config.motor_current_ma or 0)
            self.resolution.setCurrentIndex(self.resolution.findData(config.step_resolution))
            self.interpolation.setChecked(config.interpolation)
            self.unit.model().item(1).setEnabled(config.pixels_per_step is not None)
            self.unit.setCurrentIndex(self.unit.findData(config.jog_unit))
            self.distance.setValue(config.jog_distance)
            self.speed.setValue(config.motor_speed)
            self.frequency.setValue(config.repeat_hz)
            self.pixel_scale.setText(f"Pixels: {config.pixels_per_step:+.6g} px / driver step" if config.pixels_per_step is not None else "Pixels: not calibrated")
            self._effective()
        finally:
            self._syncing = False

    def _effective(self):
        suffix = {"driversteps": " steps", "pixels": " px"}[self._config.jog_unit]
        self.distance.setSuffix(suffix)
        try:
            steps = self._config.distance_steps()
            self.effective.setText(f"Effective: {steps:,} driver steps / nudge")
            self.known_distance_hint.setText(
                f"X movement for +{steps:,} driver steps. Positive pixels: image moves right; negative: left.")
            self.known_pixel_distance.setValue(steps * self._config.pixels_per_step
                                             if self._config.pixels_per_step is not None else 0.)
        except ValueError as exc:
            self.effective.setText(str(exc))

    def _unit_changed(self):
        if self._syncing:
            return
        new_unit = self.unit.currentData()
        old = self._config
        try:
            # Preserve the effective motor distance when switching units.
            scale = {"driversteps": 1.0, "pixels": abs(old.pixels_per_step or 0)}[new_unit]
            if not scale:
                raise ValueError("Calibrate this unit before selecting it")
            config = replace(old, jog_unit=new_unit, jog_distance=old.distance_steps() * scale).validated()
            self.set_config(config)
            self.config_changed.emit(config)
        except ValueError as exc:
            self.error.emit(str(exc))
            self.set_config(old)

    def _edited(self):
        if self._syncing:
            return
        old = self._config
        try:
            if self.left_channel.currentData() == self.right_channel.currentData():
                other = self.right_channel if self.sender() is self.left_channel else self.left_channel
                previous = old.left_channel if other is self.right_channel else old.right_channel
                blocked = other.blockSignals(True)
                other.setCurrentIndex(other.findData(previous))
                other.blockSignals(blocked)
            config = replace(old, host=self.host.text().strip(), port=self.port.value(),
                             stepper_uid=self.stepper_uid.text().strip(), digital_uid=self.digital_uid.text().strip(),
                             left_channel=self.left_channel.currentData(), right_channel=self.right_channel.currentData(),
                             reference_channel=self.reference_channel.currentData(),
                             active_high=self.active_high.isChecked(), invert_direction=self.invert_direction.isChecked(),
                             motor_current_ma=self.motor_current.value() or None, step_resolution=self.resolution.currentData(),
                             interpolation=self.interpolation.isChecked(), jog_unit=self.unit.currentData(),
                             jog_distance=self.distance.value(), motor_speed=self.speed.value(), repeat_hz=self.frequency.value())
            if config.step_resolution != old.step_resolution:
                config = replace(config, pixels_per_step=None,
                                 calibration_source_id="", calibration_shape=None,
                                 jog_unit="driversteps", jog_distance=1.)
            config = config.validated()
            if config.step_resolution != old.step_resolution:
                self.set_config(config)
            else:
                self._config = config
            self._effective()
            self.unit.model().item(1).setEnabled(config.pixels_per_step is not None)
            if config != old:
                self.config_changed.emit(config)
        except ValueError as exc:
            self.set_config(old)
            self.error.emit(str(exc))

    def _display_connection(self, connected):
        self.connect_button.setText("Disconnect hardware" if connected else "Connect hardware")
        for control in (self.host, self.port, self.stepper_uid, self.digital_uid):
            control.setEnabled(not connected)
        self.stop_button.setEnabled(connected)

    def set_connected(self, connected):
        self._display_connection(connected)
        self.setup_button.setEnabled(connected)
        self.arm_button.setEnabled(False)
        self.calibrate_button.setEnabled(False)
        self.use_known_distance_button.setEnabled(False)
        self.align_button.setEnabled(False)

    def display_status(self, status, context):
        # Set each action's final state once. Temporarily disabling a button
        # during a refresh cancels a mouse press before its release arrives.
        self._display_connection(status.connected)
        self.status.setText(f"{status.phase.upper()} · {status.message}")
        self.inputs.setText("   ".join(f"{i}: {'HIGH' if value else 'low'}" for i, value in enumerate(status.inputs4)))
        if status.connected:
            self.device_settings.setText(f"Device: {status.motor_current_ma} mA · 1/{status.step_resolution} step\nPosition: {status.position:,} steps · {'movement enabled' if status.armed else 'movement disabled'}")
        else:
            self.device_settings.setText("Connect to read motor settings")
        idle = status.phase in ("idle", "ready", "aligned", "stopped")
        self.setup_button.setEnabled(status.connected and status.velocity == 0 and status.remaining == 0)
        self.arm_button.setEnabled(status.connected and not status.armed)
        live = context.live_camera_ready
        self.calibrate_button.setEnabled(status.armed and idle and live)
        self.use_known_distance_button.setEnabled(
            live and (idle or status.phase in ("disconnected", "connected"))
            and status.velocity == 0 and status.remaining == 0)
        self.align_button.setEnabled(status.armed and idle and context.camera_ready and status.pixels_per_step is not None)
