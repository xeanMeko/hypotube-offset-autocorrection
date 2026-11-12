"""Remember desktop preferences across restarts when hardware is unavailable."""
from dataclasses import replace
import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtWidgets

from measure_shift.app import MainWindow, default_settings_path
from measure_shift.storage import load_profile


@pytest.fixture(scope="module")
def application():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


class DisconnectedHardware:
    def __init__(self):
        self.connects = 0

    def connect(self, *args):
        self.connects += 1
        raise RuntimeError("Hardware is unavailable in this test")

    def shutdown(self, timeout_s=2):
        pass


def pump_until(app, predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(.005)
    pytest.fail("Saved settings did not reach the expected state")


def test_edits_autosave_and_restore_across_window_restart(application, tmp_path):
    hardware = DisconnectedHardware()
    window = MainWindow(motion_hardware=hardware)
    window.show()
    path = default_settings_path()
    assert path.is_relative_to(tmp_path)
    try:
        window.motion_dock.show()
        dock = window.motion_dock
        dock.left_channel.setCurrentIndex(0)
        dock.right_channel.setCurrentIndex(3)
        dock.reference_channel.setCurrentIndex(dock.reference_channel.findData(1))
        dock.active_high.setChecked(False)
        dock.invert_direction.setChecked(False)
        dock.motor_current.setValue(900)
        dock.motor_current.editingFinished.emit()
        dock.resolution.setCurrentIndex(dock.resolution.findData(16))
        dock.distance.setValue(7)
        dock.speed.setValue(125)
        dock.frequency.setValue(4)
        window.tolerance.setValue(2.5)
        # Internal quality thresholds survive changes to the visible controls.
        window._set_config(replace(window.config, min_response=.25))
        window.scale.setCurrentIndex(window.scale.findData(.5))
        window.view_mode.setCurrentText("Absolute difference")
        window.trend_toggle.setChecked(True)
        window.dock.show()
        window.dock.rate.setValue(3)
        window.dock.tabs.setCurrentIndex(5)
        window.dock.mode.setCurrentText("Live")
        window.source_kind.setCurrentText("Basler")
        window.device.clear()
        window.device.addItem("My camera", "camera-42")
        window._profile_camera = {"fps": 21, "exposure_ms": 5.5}
        # The file must be written while the first window is still open.
        pump_until(application, lambda: path.exists() and
                   json.loads(path.read_text())["motion"]["repeat_hz"] == 4)
        config, tolerance, hz, camera, motion = load_profile(path, include_motion=True)
        assert motion["left_channel"] == 0 and motion["right_channel"] == 3
        assert motion["reference_channel"] == 1
        assert motion["motor_current_ma"] == 900 and motion["step_resolution"] == 16
        assert config.scale == .5 and tolerance == 2.5 and hz == 3
        assert camera == {"fps": 21, "exposure_ms": 5.5}
    finally:
        window.close()
        application.processEvents()

    restored = MainWindow(motion_hardware=hardware)
    restored.show()
    try:
        pump_until(application, lambda: restored.motion_controls.connect_button.isEnabled())
        dock = restored.motion_dock
        assert dock.config.left_channel == 0 and dock.config.right_channel == 3
        assert dock.config.reference_channel == 1 and dock.reference_channel.currentData() == 1
        assert not dock.config.active_high and not dock.config.invert_direction
        assert dock.config.motor_current_ma == 900 and dock.config.step_resolution == 16
        assert (dock.config.jog_distance, dock.config.motor_speed, dock.config.repeat_hz) == (7, 125, 4)
        assert restored.config.scale == .5 and restored.config.min_response == .25
        assert restored.tolerance.value() == 2.5 and restored.dock.rate.value() == 3
        assert restored.source_kind.currentText() == "Basler"
        assert restored.device.currentData() == "camera-42"
        assert restored._profile_camera == {"fps": 21, "exposure_ms": 5.5}
        assert restored.view_mode.currentText() == "Absolute difference"
        assert restored.trend_toggle.isChecked()
        assert restored.motion_dock.isVisible() and restored.dock.isVisible()
        assert restored.dock.tabs.currentIndex() == 5 and restored.dock.mode.currentText() == "Live"
        assert not restored.motion.controller.connected and not restored.motion.controller.armed
        assert hardware.connects == 2
        assert not restored.motion_controls.connect_button.isChecked()
    finally:
        restored.close()
        application.processEvents()
    # A second close without a camera connected must retain its saved settings.
    assert load_profile(path)[3] == {"fps": 21, "exposure_ms": 5.5}


def test_close_flushes_an_edit_before_autosave_timer_fires(application, tmp_path):
    path = tmp_path / "settings.json"
    window = MainWindow(settings_path=path)
    window.motion_dock.frequency.setValue(6)
    assert window.settings_timer.isActive()
    window.close()
    assert load_profile(path, include_motion=True)[4]["repeat_hz"] == 6
    assert not window.settings_timer.isActive()


def test_corrupt_settings_do_not_prevent_startup(application, tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("broken json")
    window = MainWindow(settings_path=path)
    try:
        assert "Could not restore previous settings" in window.notice.text()
        assert window.motion_dock.config.motor_current_ma == 800
    finally:
        window.close()
    assert load_profile(path, include_motion=True)[4]["motor_current_ma"] == 800


def test_unwritable_preferences_do_not_prevent_closing(application, tmp_path):
    blocked = tmp_path / "file-instead-of-directory"
    blocked.write_text("keep")
    window = MainWindow(settings_path=blocked / "settings.json")
    window.motion_dock.frequency.setValue(5)
    assert window.close()
    assert not window.motion.is_alive()
    assert "Could not save startup settings" in window.statusBar().currentMessage()
    assert blocked.read_text() == "keep"
