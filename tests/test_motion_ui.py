"""Keyboard focus, axis settings, and application wiring without real hardware."""
from dataclasses import replace
import os
import threading
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import cv2
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui, QtTest

from measure_shift.app import MainWindow
from measure_shift.motion import MotionConfig, MotionContext, MotionStatus, ReferenceCaptureRequest
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
        self.inputs = (False, False, False, False)

    def connect(self, *args):
        self.calls.append("connect")

    def read_inputs(self):
        return self.inputs

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
        pump(application, lambda: window._motion_status and window._motion_status.phase == "ready",
             timeout=3, required=True)
        if button_name == "arm_button":
            window.motion_dock.setup_button.click()
            pump(application, lambda: not window._motion_status.armed and window._motion_status.setup_applied,
                 timeout=3, required=True)
        else:
            window.motion_controls.connect_button.click()
            pump(application, lambda: not window._motion_status.connected,
                 timeout=3, required=True)
        button = (window.motion_controls.connect_button if button_name == "connect_button"
                  else window.motion_dock.arm_button)
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
    window.motion_dock.show()
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


def test_unit_switch_preserves_driver_distance_and_motor_speed(application):
    dock = MotionDock(MotionConfig(jog_distance=40, motor_speed=400, pixels_per_step=.05))
    dock.unit.setCurrentIndex(dock.unit.findData("pixels"))
    assert dock.config.jog_distance == pytest.approx(2)
    assert dock.config.motor_speed == 400
    assert dock.config.distance_steps() == 40
    assert dock.config.speed_steps() == 400
    dock.close()


def test_reference_channel_selector_supports_all_channels_and_disabled(application):
    dock = MotionDock(MotionConfig(align_channel=None))
    changes = []
    dock.config_changed.connect(lambda config: changes.append(config.reference_channel))
    assert dock.reference_channel.currentData() is None
    for channel in (*range(4), None):
        dock.reference_channel.setCurrentIndex(dock.reference_channel.findData(channel))
        dock.apply_inputs_button.click()
        assert dock.config.reference_channel == channel
    assert changes == [0, 1, 2, 3, None]
    dock.close()


def test_digital_input_captures_reference_through_worker_and_gui(axis_window, application):
    window, hardware = axis_window
    frame = FramePacket.from_array(np.full((32, 48), 80, np.uint8), "Trigger test")
    window.frames.publish(frame)
    pump(application, lambda: window.latest_frame is frame, timeout=2, required=True)
    window.motion_dock.reference_channel.setCurrentIndex(window.motion_dock.reference_channel.findData(0))
    window.motion_dock.apply_inputs_button.click()
    pump(application, lambda: window._motion_status.config_revision == window._motion_config_revision,
         timeout=2, required=True)
    assert window.reference is None
    hardware.inputs = (True, False, False, False)
    pump(application, lambda: window.reference is not None, timeout=2, required=True)
    assert window.reference.frame.frame_id == frame.frame_id
    reference_id = window.reference.reference_id
    window.capture_reference()
    assert window.reference.reference_id != reference_id
    reference_id = window.reference.reference_id
    pump(application, timeout=.1)
    hardware.inputs = (False, False, False, False)
    pump(application, lambda: not window._motion_status.inputs4[0], timeout=.3, required=True)
    hardware.inputs = (True, False, False, False)
    pump(application, lambda: window._motion_status.inputs4[0], timeout=.3, required=True)
    assert window.reference.reference_id == reference_id
    pump(application, timeout=1.05)
    assert window.reference.reference_id == reference_id
    hardware.inputs = (False, False, False, False)
    pump(application, lambda: not window._motion_status.inputs4[0], timeout=.3, required=True)
    hardware.inputs = (True, False, False, False)
    pump(application, lambda: window.reference.reference_id != reference_id, timeout=2, required=True)
    assert hardware.moves == []


def test_gui_limits_delayed_digital_captures_and_discards_stale_requests(application, monkeypatch):
    import measure_shift.app as app_module

    window = MainWindow(motion_hardware=FakeHardware())
    window.timer.stop()
    window.motion.stop_worker()
    window.motion.join(timeout=2)
    now = [10.]
    monkeypatch.setattr(app_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    captures = []
    monkeypatch.setattr(window, "capture_reference", lambda: captures.append(now[0]))
    config = MotionConfig(reference_channel=0)
    window.motion_dock.set_config(config)
    status = MotionStatus(connected=True, config=config)

    def publish(triggered_at, **kwargs):
        request = ReferenceCaptureRequest(0, triggered_at, 0)
        window.motion.output.publish(replace(status, reference_capture=request, **kwargs))
        window._poll_motion()

    try:
        publish(1.)
        assert captures == [10.]
        now[0] = 10.2
        publish(2.)  # Input edges were a second apart; GUI handling is closer.
        assert captures == [10.]
        now[0] = 11.
        window._poll_motion()  # A suppressed request must not be replayed.
        assert captures == [10.]
        publish(3.)
        assert captures == [10., 11.]
        now[0] = 12.
        publish(4., connected=False)
        assert captures == [10., 11.]
        window._motion_config_revision = 1
        publish(5.)  # An edge from the previous channel configuration is stale.
        assert captures == [10., 11.]
    finally:
        window.close()
        application.processEvents()


def test_main_calibration_button_uses_the_test_move_from_axis_settings(application, monkeypatch):
    window = MainWindow(motion_hardware=FakeHardware())
    requested = []
    monkeypatch.setattr(window.motion, "calibrate", requested.append)
    try:
        window.motion_controls.calibrate_button.setEnabled(True)
        window.motion_controls.calibrate_button.click()
        assert requested == [100]
        window.motion_dock.test_steps.setValue(-250)
        assert "-250 driver steps" in window.motion_controls.calibrate_button.toolTip()
        window.motion_controls.calibrate_button.click()
        assert requested == [100, -250]
    finally:
        window.close()
        application.processEvents()


def test_main_axis_controls_work_with_settings_closed(application):
    hardware = FakeHardware()
    window = MainWindow(motion_hardware=hardware)
    window.show()
    try:
        controls = window.motion_controls
        application.processEvents()
        assert not window.motion_dock.isVisible()
        toolbar = window.main_toolbar
        assert window.findChildren(QtWidgets.QToolBar) == [toolbar]
        assert window.toolBarArea(toolbar) == QtCore.Qt.ToolBarArea.TopToolBarArea
        assert not window.toolBarBreak(toolbar)
        assert not window.tools_menu.isVisible()
        assert window.motion_dock.toggleViewAction() in window.tools_menu.actions()
        window.resize(1100, 760)
        application.processEvents()
        for button in (controls.connect_button, controls.calibrate_button,
                       controls.align_button, controls.stop_button):
            assert button.isVisible()
            assert toolbar.isAncestorOf(button)
            assert not button.visibleRegion().isEmpty()
        assert not window.tools_button.visibleRegion().isEmpty()
        assert not controls.calibrate_button.isEnabled()
        assert not controls.align_button.isEnabled()
        pump(application, lambda: window._motion_status and window._motion_status.phase == "ready",
             timeout=3, required=True)
        assert controls.connect_button.isCheckable()
        assert controls.connect_button.isChecked()
        assert controls.connect_button.text() == "Disconnect hardware"
        connected_size = controls.connect_button.size()
        assert connected_size.width() >= controls.connect_button.sizeHint().width()
        assert hardware.calls.count("connect") == 1
        assert hardware.state.enabled and hardware.moves == []
        assert controls.stop_button.isEnabled()
        assert not controls.stop_button.visibleRegion().isEmpty()
        assert not window.tools_button.visibleRegion().isEmpty()
        assert not controls.calibrate_button.isEnabled()  # No live camera.
        assert not controls.align_button.isEnabled()
        assert "READY" in controls.status.text()
        controls.stop_button.click()
        pump(application, lambda: "stop" in hardware.calls, timeout=3, required=True)
        controls.connect_button.click()
        pump(application, lambda: not window._motion_status.connected, timeout=3, required=True)
        assert controls.connect_button.text() == "Connect hardware"
        assert controls.connect_button.size() == connected_size
        assert not controls.connect_button.isChecked()
        assert not hardware.state.enabled
        assert not controls.stop_button.isEnabled()
        controls.connect_button.click()
        pump(application, lambda: window._motion_status.phase == "ready", timeout=3, required=True)
        assert controls.connect_button.isChecked()
        assert controls.connect_button.text() == "Disconnect hardware"
        assert controls.connect_button.size() == connected_size
        assert hardware.calls.count("connect") == 2
        assert hardware.state.enabled and hardware.moves == []
        assert not window.motion_dock.isVisible()
        window.tools_menu.popup(window.tools_button.mapToGlobal(QtCore.QPoint(0, window.tools_button.height())))
        application.processEvents()
        assert window.tools_menu.isVisible()
        window.motion_dock.toggleViewAction().trigger()
        window.tools_menu.close()
        application.processEvents()
        assert window.motion_dock.isVisible()
        assert not window.tools_menu.isVisible()
    finally:
        window.close()
        application.processEvents()


def test_startup_connection_keeps_toggle_pending_until_hardware_responds(application):
    opening, release = threading.Event(), threading.Event()

    class SlowHardware(FakeHardware):
        def connect(self, *args):
            opening.set()
            if not release.wait(3):
                raise RuntimeError("Test connection timed out")
            super().connect(*args)

    hardware = SlowHardware()
    window = MainWindow(motion_hardware=hardware)
    try:
        pump(application, opening.is_set, timeout=2, required=True)
        pump(application, timeout=.15)
        button = window.motion_controls.connect_button
        assert button.isChecked() and not button.isEnabled()
        assert not hardware.moves
        release.set()
        pump(application, lambda: button.isEnabled(), timeout=3, required=True)
        assert button.isChecked() and window._motion_status.armed
        assert hardware.calls.count("connect") == 1 and not hardware.moves
    finally:
        release.set()
        window.close()
        application.processEvents()


def test_failed_startup_connection_can_be_retried_and_connection_loss_clears_toggle(application):
    class UnavailableHardware(FakeHardware):
        unavailable = True

        def connect(self, *args):
            super().connect(*args)
            if self.unavailable:
                raise RuntimeError("Hardware unavailable")

        def read_state(self):
            if self.unavailable:
                raise RuntimeError("Connection lost")
            return super().read_state()

    hardware = UnavailableHardware()
    window = MainWindow(motion_hardware=hardware)
    button = window.motion_controls.connect_button
    try:
        pump(application, lambda: button.isEnabled(), timeout=3, required=True)
        assert not button.isChecked()
        assert "Hardware unavailable" in window.motion_controls.status.text()
        pump(application, timeout=.15)
        assert hardware.calls.count("connect") == 1
        hardware.unavailable = False
        button.click()
        pump(application, lambda: window._motion_status.phase == "ready", timeout=3, required=True)
        assert button.isChecked() and hardware.state.enabled
        hardware.unavailable = True
        pump(application, lambda: not button.isChecked(), timeout=3, required=True)
        assert "Connection lost" in window.motion_controls.status.text()
        assert not hardware.state.enabled and not hardware.moves
        assert hardware.calls.count("connect") == 2
    finally:
        window.close()
        application.processEvents()


def test_startup_connects_using_restored_axis_settings(application, tmp_path):
    from measure_shift.models import MatchConfig
    from measure_shift.storage import save_profile

    path = tmp_path / "startup.json"
    config = MotionConfig(host="saved-host", port=4224, stepper_uid="saved-stepper",
                          digital_uid="saved-input", motor_current_ma=900, step_resolution=16)
    save_profile(path, MatchConfig(), 1, 1, motion=config.to_dict())

    class RecordingHardware(FakeHardware):
        def connect(self, *args):
            self.connection_args = args
            super().connect(*args)

    hardware = RecordingHardware()
    window = MainWindow(motion_hardware=hardware, settings_path=path)
    try:
        pump(application, lambda: window._motion_status and window._motion_status.phase == "ready",
             timeout=3, required=True)
        assert hardware.connection_args == ("saved-host", 4224, "saved-stepper", "saved-input")
        assert hardware.state.motor_current_ma == 900 and hardware.state.step_resolution == 16
        assert hardware.state.enabled and hardware.moves == []
        assert hardware.calls.count("connect") == 1
        assert window.motion_controls.connect_button.isChecked()
    finally:
        window.close()
        application.processEvents()


@pytest.mark.parametrize("steps,pixels", [(400, 20), (400, -20), (1, 49)])
def test_known_pixel_nudge_calibrates_without_motor_movement_and_is_saved(
        application, monkeypatch, tmp_path, steps, pixels):
    from measure_shift.storage import load_profile

    hardware = FakeHardware()
    path = tmp_path / "known-distance.json"
    window = MainWindow(motion_hardware=hardware, settings_path=path)
    pump(application, lambda: window._motion_status and window._motion_status.phase == "ready",
         timeout=3, required=True)
    window.motion_controls.connect_button.click()
    pump(application, lambda: not window._motion_status.connected, timeout=3, required=True)
    connects = hardware.calls.count("connect")
    context = MotionContext(source_id="ids:serial:known", source_kind="IDS", shape=(32, 48),
                            config_id=window.config.config_id, connected=True)
    monkeypatch.setattr(window, "_axis_context", lambda: context)
    try:
        dock = window.motion_dock
        dock.distance.setValue(steps)
        pump(application, lambda: dock.use_known_distance_button.isEnabled(), timeout=2, required=True)
        assert f"+{steps} driver steps" in dock.known_distance_hint.text()
        dock.known_pixel_distance.setValue(pixels)
        dock.use_known_distance_button.click()
        pump(application, lambda: window._motion_status.config_revision == window._motion_config_revision
             and window._motion_status.config.pixels_per_step == pixels / steps, timeout=2, required=True)
        assert dock.unit.currentData() == "pixels"
        assert dock.distance.value() == abs(pixels)
        assert dock.config.distance_steps() == steps
        assert dock.config.calibration_source_id == context.source_id
        assert dock.config.calibration_shape == context.shape
        assert dock.config.speed_steps() == 200
        assert window.reference is None
        assert not hardware.moves and hardware.calls.count("connect") == connects
        assert not hardware.state.enabled
        dock.distance.setValue(abs(pixels) * 2)
        assert dock.config.distance_steps() == steps * 2
        assert dock.known_pixel_distance.value() == pixels * 2
    finally:
        window.close()
        application.processEvents()
    saved = MotionConfig.from_dict(load_profile(path, include_motion=True)[4])
    assert saved.pixels_per_step == pixels / steps
    assert saved.jog_distance == abs(pixels) * 2 and saved.jog_unit == "pixels"
    assert saved.calibration_source_id == context.source_id


def test_known_pixel_distance_rejects_zero_and_requires_a_live_camera(application, monkeypatch):
    window = MainWindow(motion_hardware=FakeHardware())
    try:
        before = window.motion_dock.config
        assert not window.motion_dock.use_known_distance_button.isEnabled()
        window._set_known_pixel_distance(20)
        assert "live IDS or Basler camera" in window.notice.text()
        assert window.motion_dock.config == before
        context = MotionContext(source_id="ids:known", source_kind="IDS", shape=(32, 48),
                                config_id=window.config.config_id, connected=True)
        monkeypatch.setattr(window, "_axis_context", lambda: context)
        window._set_known_pixel_distance(0)
        assert "nonzero" in window.notice.text()
        assert window.motion_dock.config == before
        assert not window.motion.hardware.moves
    finally:
        window.close()
        application.processEvents()


def test_resolution_change_clears_scales_and_visible_units(application):
    dock = MotionDock(MotionConfig(jog_unit="pixels", jog_distance=2, motor_speed=400,
                                   pixels_per_step=.05))
    dock.resolution.setCurrentIndex(dock.resolution.findData(16))
    assert dock.config.pixels_per_step is None
    assert dock.unit.currentData() == "driversteps"
    dock.frequency.setValue(3)
    assert dock.config.pixels_per_step is None
    dock.close()


def test_digital_inputs_apply_together_and_allow_complete_channel_swaps(application):
    dock = MotionDock()
    changes = []
    dock.config_changed.connect(changes.append)
    assert dock.align_channel.currentData() == 3
    assert not dock.apply_inputs_button.isEnabled()
    old = dock.config
    # Intermediate assignments can overlap without changing the live wiring.
    dock.left_channel.setCurrentIndex(1)
    dock.right_channel.setCurrentIndex(3)
    dock.reference_channel.setCurrentIndex(dock.reference_channel.findData(0))
    dock.align_channel.setCurrentIndex(dock.align_channel.findData(2))
    dock.active_high.setChecked(False)
    dock.invert_direction.setChecked(False)
    assert dock.config == old and changes == []
    assert dock.apply_inputs_button.isEnabled()
    dock.apply_inputs_button.click()
    assert len(changes) == 1
    assert (dock.config.left_channel, dock.config.right_channel,
            dock.config.reference_channel, dock.config.align_channel) == (1, 3, 0, 2)
    assert not dock.config.active_high and not dock.config.invert_direction
    assert not dock.apply_inputs_button.isEnabled()
    dock.close()


def test_invalid_channel_mapping_stays_pending_until_corrected(application):
    dock = MotionDock()
    errors = []
    changes = []
    dock.error.connect(errors.append)
    dock.config_changed.connect(changes.append)
    old = dock.config
    dock.left_channel.setCurrentIndex(1)
    dock.apply_inputs_button.click()
    assert errors and "different digital inputs" in errors[-1]
    assert dock.config == old and changes == []
    assert dock.left_channel.currentData() == 1
    assert dock.apply_inputs_button.isEnabled()
    dock.right_channel.setCurrentIndex(2)
    dock.align_channel.setCurrentIndex(dock.align_channel.findData(2))
    dock.apply_inputs_button.click()
    assert "Align X must use a different digital input" in errors[-1]
    assert dock.config == old and changes == []
    dock.align_channel.setCurrentIndex(dock.align_channel.findData(None))
    dock.apply_inputs_button.click()
    assert len(changes) == 1 and dock.config.align_channel is None
    dock.close()


def test_other_edits_and_status_sync_preserve_pending_input_mapping(application):
    dock = MotionDock()
    errors = []
    dock.error.connect(errors.append)
    dock.left_channel.setCurrentIndex(1)
    dock.right_channel.setCurrentIndex(2)
    dock.frequency.setValue(3)
    assert dock.config.repeat_hz == 3
    assert (dock.config.left_channel, dock.config.right_channel) == (2, 1)
    dock.resolution.setCurrentIndex(dock.resolution.findData(16))
    dock.distance.setValue(.5)
    assert errors and dock.distance.value() == 1
    assert dock.config.jog_distance == 1
    dock.set_config(replace(dock.config, pixels_per_step=2), preserve_input_edits=True)
    dock.unit.setCurrentIndex(dock.unit.findData("pixels"))
    assert dock.left_channel.currentData() == 1 and dock.right_channel.currentData() == 2
    assert (dock.config.left_channel, dock.config.right_channel) == (2, 1)
    dock.apply_inputs_button.click()
    assert dock.config.left_channel == 1 and dock.config.right_channel == 2
    dock.left_channel.setCurrentIndex(0)
    dock.set_config(MotionConfig())  # Loading a profile replaces pending edits.
    assert dock.left_channel.currentData() == 2 and not dock.apply_inputs_button.isEnabled()
    dock.close()


def test_zero_calibration_reports_error_and_closing_joins_motion(axis_window, application):
    window, hardware = axis_window
    window._calibrate_motion(0)
    assert "nonzero" in window.notice.text()
    window.close()
    assert not window.motion.is_alive()
    assert hardware.calls[-1] == "shutdown"


@pytest.mark.parametrize("uncertain_large_offset", [False, True])
@pytest.mark.parametrize("calibration_reference", ["target", "unrelated", "none"])
def test_real_app_calibration_and_alignment_with_simulated_axis(
        axis_window, application, monkeypatch, uncertain_large_offset, calibration_reference):
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
    if calibration_reference != "none":
        saved_image = np.zeros_like(reference) if calibration_reference == "unrelated" else reference
        window.set_reference(FramePacket.from_array(saved_image))
        pump(application, lambda: window.last_pair is not None, timeout=3, required=True)
    saved_reference = window.reference
    if calibration_reference == "unrelated":
        assert not window.last_pair.result.usable
    assert not window.motion_dock.isVisible()
    assert window.motion_controls.calibrate_button.isVisible()
    pump(application, lambda: window.motion_controls.calibrate_button.isEnabled(),
         timeout=3, required=True)
    window.motion_dock.distance.setValue(6)
    window.motion_dock.test_steps.setValue(8)
    window.motion_controls.calibrate_button.click()
    pump(application, lambda: window.motion_dock.config.pixels_per_step is not None,
         timeout=5, required=True)
    assert window.motion_dock.config.pixels_per_step == pytest.approx(.5, abs=.04)
    assert window.motion_dock.unit.currentData() == "pixels"
    assert window.motion_dock.distance.suffix() == " px"
    assert window.motion_dock.config.calibration_source_id == "ids:serial:fake-serial"
    assert "px / driver step" in window.motion_dock.pixel_scale.text()
    assert window.reference is saved_reference
    if calibration_reference == "none":
        assert not window.motion_controls.align_button.isEnabled()
    if calibration_reference != "target":
        window.set_reference(FramePacket.from_array(reference))
        pump(application, lambda: window.last_pair is not None and window.last_pair.result.usable,
             timeout=3, required=True)
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
    pump(application, lambda: window.motion_controls.align_button.isEnabled(), timeout=2, required=True)
    if calibration_reference == "none":
        hardware.inputs = (False, False, False, True)
    else:
        window.motion_controls.align_button.click()
    pump(application, lambda: "X aligned" in window._motion_status.message, timeout=8, required=True)
    assert abs(window.last_pair.result.dx) <= window.tolerance.value()
    assert window.last_pair.result.dy > 2
    assert window.state.text() == "OUTSIDE TOLERANCE"
    assert camera.freshen_count >= 4
    assert not window.motion_dock.isVisible()


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
