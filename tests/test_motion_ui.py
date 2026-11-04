"""Keyboard focus, axis settings, and application wiring without real hardware."""
from dataclasses import replace
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import cv2
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui, QtTest

from measure_shift.app import MainWindow
from measure_shift.motion import MotionConfig
from measure_shift.motion_hardware import HardwareState
from measure_shift.motion_widgets import AxisKeyboardFilter, MotionDock
from measure_shift.models import FramePacket


@pytest.fixture(scope="module")
def application():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


def pump(app, predicate=lambda: False, timeout=.1, required=False):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(.002)
    if required:
        pytest.fail("Motion UI did not reach the expected state")


class FakeHardware:
    def __init__(self):
        self.state = HardwareState(motor_current_ma=800)
        self.moves = []
        self.calls = []

    def connect(self, *args):
        self.calls.append("connect")

    def read_inputs(self):
        return False, False, False, False

    def read_state(self):
        return self.state

    def configure(self, motor_current_ma, step_resolution=8, interpolation=True):
        self.state = replace(self.state, motor_current_ma=motor_current_ma,
                             step_resolution=step_resolution, interpolation=interpolation, enabled=False)

    def enable(self):
        self.state = replace(self.state, enabled=True)

    def disable(self):
        self.state = replace(self.state, enabled=False)

    def move_steps(self, steps, velocity):
        self.moves.append((steps, velocity))
        self.state = replace(self.state, position=self.state.position + steps)

    def stop(self):
        self.calls.append("stop")

    def shutdown(self, timeout_s=2.):
        self.calls.append("shutdown")
        self.state = replace(self.state, enabled=False)


@pytest.fixture
def axis_window(application):
    hardware = FakeHardware()
    window = MainWindow(motion_hardware=hardware)
    window.show()
    window.activateWindow()
    window.motion_dock.show()
    window.motion_dock.connect_button.click()
    pump(application, lambda: window._motion_status and window._motion_status.phase == "ready", timeout=3, required=True)
    window.live_pane.plot.setFocus()
    window.activateWindow()
    application.processEvents()
    yield window, hardware
    window.close()
    application.processEvents()
    assert not window.motion.is_alive()


@pytest.mark.parametrize("button_name", ["connect_button", "arm_button"])
def test_connect_and_enable_mouse_clicks_survive_status_refresh(application, button_name):
    hardware = FakeHardware()
    window = MainWindow(motion_hardware=hardware)
    window.show()
    window.activateWindow()
    window.motion_dock.show()
    try:
        assert window.motion_dock.motor_current.value() == 800
        assert window.motion_dock.left_channel.currentData() == 2
        assert window.motion_dock.right_channel.currentData() == 1
        if button_name == "arm_button":
            window.motion_dock.connect_button.click()
            pump(application, lambda: window._motion_status and window._motion_status.phase == "ready",
                 timeout=3, required=True)
            window.motion_dock.setup_button.click()
            pump(application, lambda: not window._motion_status.armed and window._motion_status.setup_applied,
                 timeout=3, required=True)
        button = getattr(window.motion_dock, button_name)
        QtTest.QTest.mousePress(button, QtCore.Qt.MouseButton.LeftButton)
        # A normal human click spans multiple status timer updates.
        pump(application, timeout=.15)
        assert button.isDown()
        QtTest.QTest.mouseRelease(button, QtCore.Qt.MouseButton.LeftButton)
        pump(application, lambda: window._motion_status and window._motion_status.armed and
             window._motion_status.phase == "ready", timeout=3, required=True)
        assert hardware.state.enabled
        assert hardware.state.motor_current_ma == 800
        assert window._motion_status.setup_applied
        assert hardware.moves == []
    finally:
        window.close()
        application.processEvents()


def test_keyboard_press_executes_one_move_without_replaying_after_release(axis_window, application):
    window, hardware = axis_window
    QtTest.QTest.keyPress(window.live_pane.plot, QtCore.Qt.Key.Key_Left)
    pump(application, lambda: len(hardware.moves) == 1, timeout=2, required=True)
    QtTest.QTest.keyRelease(window.live_pane.plot, QtCore.Qt.Key.Key_Left)
    assert hardware.moves == [(1, 200)]
    pump(application, timeout=.6)
    assert len(hardware.moves) == 1


def test_keyboard_repeat_uses_ui_frequency_and_release_finishes(axis_window, application):
    window, hardware = axis_window
    window.motion_dock.frequency.setValue(10)
    window.motion_dock.distance.setValue(3)
    window.motion_dock.speed.setValue(400)
    window.live_pane.plot.setFocus()
    QtTest.QTest.keyPress(window.live_pane.plot, QtCore.Qt.Key.Key_Right)
    pump(application, lambda: len(hardware.moves) >= 3, timeout=2, required=True)
    QtTest.QTest.keyRelease(window.live_pane.plot, QtCore.Qt.Key.Key_Right)
    pump(application, timeout=.08)
    count = len(hardware.moves)
    pump(application, timeout=.2)
    assert len(hardware.moves) == count
    assert all(move == (-3, 400) for move in hardware.moves)


def test_focus_loss_without_release_clears_keyboard(axis_window, application):
    window, hardware = axis_window
    QtTest.QTest.keyPress(window.live_pane.plot, QtCore.Qt.Key.Key_Right)
    pump(application, lambda: bool(hardware.moves), timeout=2, required=True)
    QtWidgets.QApplication.sendEvent(window, QtCore.QEvent(QtCore.QEvent.Type.WindowDeactivate))
    assert not window.axis_keyboard.right
    pump(application, timeout=.6)
    assert len(hardware.moves) == 1


def test_settings_fields_menus_and_modified_arrows_do_not_nudge(axis_window, application):
    window, hardware = axis_window
    window.motion_dock.distance.setFocus()
    QtTest.QTest.keyClick(window.motion_dock.distance, QtCore.Qt.Key.Key_Right)
    window.live_pane.plot.setFocus()
    QtTest.QTest.keyClick(window.live_pane.plot, QtCore.Qt.Key.Key_Left,
                          QtCore.Qt.KeyboardModifier.ControlModifier)
    menu = QtWidgets.QMenu(window)
    menu.addAction("Example")
    menu.popup(window.mapToGlobal(QtCore.QPoint(20, 20)))
    application.processEvents()
    QtTest.QTest.keyClick(menu, QtCore.Qt.Key.Key_Right)
    menu.close()
    pump(application, timeout=.15)
    assert hardware.moves == []


def test_os_repeat_events_do_not_release_or_duplicate_real_key(application):
    window = QtWidgets.QWidget()
    window.show()
    window.activateWindow()
    window.setFocus()
    keyboard = AxisKeyboardFilter(window)
    seen = []
    keyboard.changed.connect(lambda left, right: seen.append((left, right)))
    application.installEventFilter(keyboard)
    try:
        application.processEvents()
        QtTest.QTest.keyPress(window, QtCore.Qt.Key.Key_Left)
        for event_type in (QtCore.QEvent.Type.KeyRelease, QtCore.QEvent.Type.KeyPress):
            event = QtGui.QKeyEvent(event_type, QtCore.Qt.Key.Key_Left,
                                   QtCore.Qt.KeyboardModifier.NoModifier, "", True)
            application.sendEvent(window, event)
        assert seen == [(True, False)]
        assert keyboard.left
        QtTest.QTest.keyRelease(window, QtCore.Qt.Key.Key_Left)
        assert seen == [(True, False), (False, False)]
    finally:
        application.removeEventFilter(keyboard)
        window.close()


def test_unit_switch_preserves_driver_distance_and_speed(application):
    dock = MotionDock(MotionConfig(jog_distance=40, jog_speed=400, pixels_per_step=.05))
    dock.unit.setCurrentIndex(dock.unit.findData("pixels"))
    assert dock.config.jog_distance == pytest.approx(2)
    assert dock.config.jog_speed == pytest.approx(20)
    assert dock.config.distance_steps() == 40
    assert dock.config.speed_steps() == 400
    dock.close()


def test_default_calibration_move_uses_100_steps(application):
    dock = MotionDock()
    requested = []
    dock.calibrate_requested.connect(requested.append)
    dock.calibrate_button.setEnabled(True)
    dock.calibrate_button.click()
    assert requested == [100]
    dock.close()


def test_resolution_change_clears_scales_and_visible_units(application):
    dock = MotionDock(MotionConfig(jog_unit="pixels", jog_distance=2, jog_speed=20,
                                   pixels_per_step=.05))
    dock.resolution.setCurrentIndex(dock.resolution.findData(16))
    assert dock.config.pixels_per_step is None
    assert dock.unit.currentData() == "driversteps"
    dock.frequency.setValue(3)
    assert dock.config.pixels_per_step is None
    dock.close()


def test_rejected_edit_reverts_and_channel_swaps_are_atomic(application):
    dock = MotionDock()
    errors = []
    dock.error.connect(errors.append)
    dock.speed.setValue(100000)
    assert errors and dock.speed.value() == 200
    assert dock.config.jog_speed == 200
    dock.left_channel.setCurrentIndex(1)
    assert dock.config.left_channel == 1 and dock.config.right_channel == 2
    assert dock.right_channel.currentData() == 2
    dock.close()


def test_zero_calibration_reports_error_and_closing_joins_motion(axis_window, application):
    window, hardware = axis_window
    window._calibrate_motion(0)
    assert "nonzero" in window.notice.text()
    window.close()
    assert not window.motion.is_alive()
    assert hardware.calls[-1] == "shutdown"


@pytest.mark.parametrize("uncertain_large_offset", [False, True])
def test_real_app_calibration_and_alignment_with_simulated_axis(
        axis_window, application, monkeypatch, uncertain_large_offset):
    """Exercise camera barriers, measurement threads, UI config sync, and motor feedback together."""
    import measure_shift.app as app_module
    window, hardware = axis_window
    rng = np.random.default_rng(713)
    reference = cv2.GaussianBlur(rng.integers(20, 235, (128, 160), dtype=np.uint8), (3, 3), 0)

    class Camera:
        name = "Simulated feedback camera"
        device_identity = "fake-serial"
        ended = False
        freshen_count = 0

        def open(self): pass
        def close(self): pass
        def set_parameters(self, **kwargs): pass
        def controls(self):
            return {"fps": {"min": 1, "max": 100, "value": 100},
                    "exposure_ms": {"min": 1, "max": 100, "value": 5}}

        def read(self, timeout_ms=200):
            time.sleep(.01)
            dx = 12 + hardware.state.position * .5
            return cv2.warpAffine(reference, np.float32([[1, 0, dx], [0, 1, 4]]), (160, 128))

        def discard_pending_frames(self):
            self.freshen_count += 1

    camera = Camera()
    monkeypatch.setattr(app_module, "create_source", lambda *args, **kwargs: camera)
    window.start_source("IDS", identifier="simulated")
    pump(application, lambda: window.latest_frame is not None, timeout=3, required=True)
    window.set_reference(FramePacket.from_array(reference))
    pump(application, lambda: window.last_pair is not None and window.motion_dock.calibrate_button.isEnabled(),
         timeout=3, required=True)
    window.motion_dock.distance.setValue(6)
    window.motion_dock.test_steps.setValue(8)
    window.motion_dock.calibrate_button.click()
    pump(application, lambda: window.motion_dock.config.pixels_per_step is not None,
         timeout=5, required=True)
    assert window.motion_dock.config.pixels_per_step == pytest.approx(.5, abs=.04)
    assert window.motion_dock.unit.currentData() == "pixels"
    assert window.motion_dock.distance.suffix() == " px"
    assert window.motion_dock.config.calibration_source_id == "ids:serial:fake-serial"
    if uncertain_large_offset:
        from measure_shift.engine import MeasurementEngine
        measure = MeasurementEngine.measure

        def coarse_uncertain(self, frame, config):
            result = measure(self, frame, config)
            if result.usable and max(abs(result.dx), abs(result.dy)) > 10:
                return replace(result, validity="uncertain", reasons=("Weak match response",))
            return result

        monkeypatch.setattr(MeasurementEngine, "measure", coarse_uncertain)
        pump(application, lambda: window.last_pair.result.validity == "uncertain",
             timeout=2, required=True)
    pump(application, lambda: window.motion_dock.align_button.isEnabled(), timeout=2, required=True)
    window.motion_dock.align_button.click()
    pump(application, lambda: "X aligned" in window._motion_status.message, timeout=8, required=True)
    assert abs(window.last_pair.result.dx) <= window.tolerance.value()
    assert window.last_pair.result.dy > 2
    assert window.state.text() == "OUTSIDE TOLERANCE"
    assert camera.freshen_count >= 4


def test_loading_profile_disconnects_hardware_without_activation(axis_window, application, monkeypatch, tmp_path):
    from measure_shift.storage import save_profile
    window, hardware = axis_window
    path = tmp_path / "axis-settings.json"
    config = replace(window.motion_dock.config, repeat_hz=4, jog_distance=2)
    save_profile(path, window.config, 1, 1, motion=config.to_dict())
    monkeypatch.setattr(window, "_choose_file", lambda *args: str(path))
    connects = hardware.calls.count("connect")
    window.load_settings()
    pump(application, lambda: not window._motion_status.connected and
         window._motion_status.config_revision == window._motion_config_revision, timeout=3, required=True)
    assert window.motion_dock.config.repeat_hz == 4
    assert not window._motion_status.armed
    assert not hardware.state.enabled
    assert hardware.calls.count("connect") == connects
