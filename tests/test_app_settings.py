"""Remember desktop preferences across restarts when hardware is unavailable."""
from dataclasses import replace
import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtWidgets

from measure_shift.app import MainWindow, default_settings_path
from measure_shift.models import MatchConfig
from measure_shift.storage import load_profile, save_profile


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
        dock.align_channel.setCurrentIndex(dock.align_channel.findData(2))
        dock.active_high.setChecked(False)
        dock.apply_inputs_button.click()
        dock.pulse_ms.setValue(120)
        dock.release_ms.setValue(140)
        dock.frequency.setValue(4)
        dock.test_presses.setValue(-250)
        window.tolerance.setValue(2.5)
        # Internal quality thresholds survive changes to the visible controls.
        window._set_config(replace(window.config, min_response=.25, scale=.5))
        window.display_scale.setCurrentText("1:6")
        window.view_mode.select("Absolute difference")
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
        assert motion["align_channel"] == 2
        assert motion["pulse_ms"] == 120 and motion["release_ms"] == 140
        assert motion["calibration_presses"] == -250
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
        assert dock.config.align_channel == 2 and dock.align_channel.currentData() == 2
        assert not dock.config.active_high
        assert dock.config.pulse_ms == 120 and dock.config.release_ms == 140
        assert dock.config.repeat_hz == 4
        assert dock.config.calibration_presses == -250 and dock.test_presses.value() == -250
        assert "250 Tastendrücke nach links" in restored.motion_dock.calibrate_button.toolTip()
        assert restored.config.scale == .5 and restored.config.min_response == .25
        assert restored.tolerance.value() == 2.5 and restored.dock.rate.value() == 3
        assert restored.source_kind.currentText() == "Basler"
        assert restored.device.currentData() == "camera-42"
        assert restored._profile_camera == {"fps": 21, "exposure_ms": 5.5}
        assert restored.view_mode.currentData() == "Absolute difference"
        assert restored.display_scale.currentText() == "1:6"
        assert restored.reference_pane.display_scale == pytest.approx(1 / 6)
        assert restored.live_pane.display_scale == pytest.approx(1 / 6)
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


@pytest.mark.parametrize("saved_language", [None, "fr"])
def test_old_settings_open_in_german_and_remember_english(application, tmp_path, monkeypatch, saved_language):
    path = tmp_path / "startup.json"
    preferences = {"source_kind": "Images", "view_mode": "Absolute difference", "analytics_mode": "Frozen"}
    if saved_language is not None:
        preferences["language"] = saved_language
    save_profile(path, MatchConfig(), 1, 1, ui=preferences)
    window = MainWindow(settings_path=path)
    try:
        assert window.language_actions["de"].isChecked()
        assert window.source_kind.currentText() == "Bilder"
        assert window.view_mode.currentText() == "Absolute Differenz"
        assert window.dock.mode.currentText() == "Eingefroren"
        settings = next(action.menu() for action in window.menuBar().actions() if action.text() == "Einstellungen")
        submenu = settings.actions()[0].menu()
        assert submenu.title() == "Sprache"
        assert [action.text() for action in submenu.actions()] == ["Deutsch", "English"]
        submenu.actions()[1].trigger()
        assert settings.title() == "Settings" and submenu.title() == "Language"
        assert window.source_kind.currentText() == "Images"
        assert window.view_mode.currentText() == "Absolute difference"
        profile = tmp_path / "import.json"
        save_profile(profile, MatchConfig(), 2, 1, ui={"language": "de"})
        monkeypatch.setattr(window, "_choose_file", lambda *args: str(profile))
        window.load_settings()
        assert window.ui.code == "en" and window.tolerance.value() == 2
    finally:
        window.close()
    restored = MainWindow(settings_path=path)
    try:
        assert restored.language_actions["en"].isChecked()
        assert restored.view_mode.currentData() == "Absolute difference"
        assert json.loads(path.read_text())["ui"]["language"] == "en"
    finally:
        restored.close()


def test_switch_language_during_live_measurement(application, tmp_path):
    hardware = DisconnectedHardware()
    window = MainWindow(motion_hardware=hardware, settings_path=tmp_path / "session.json")
    try:
        window.show()
        window.start_source("demo", auto_reference=True)
        pump_until(application, lambda: window.last_pair is not None and hardware.connects == 1)
        window.capture_inspection()
        pump_until(application, lambda: window.last_diagnostic is not None)
        page = window.dock.pages["Image"]
        reference, capture = window.reference, window.capture_worker
        snapshot, roi = page.snapshot, page.line_roi
        window.display_scale.setCurrentText("1:2")
        for language, tab, axis in (("en", "Image", "Distance along line"), ("de", "Bild", "Abstand entlang der Linie")):
            window.language_actions[language].trigger()
            assert window.dock.tabs.tabText(0) == tab
            assert page.profile.getAxis("bottom").labelText == axis
            assert window.reference is reference and window.capture_worker is capture
            assert page.snapshot is snapshot and page.line_roi is roi
            assert window.dock.mode.currentData() == "Frozen"
            assert window.live_pane.display_scale == .5
        last_frame = window.last_pair.frame.frame_id
        pump_until(application, lambda: window.last_pair.frame.frame_id != last_frame)
        assert hardware.connects == 1
    finally:
        window.close()


@pytest.mark.parametrize("language", ["de", "en"])
def test_controls_and_plot_fonts_follow_their_window(application, tmp_path, language):
    window = MainWindow(settings_path=tmp_path / "sizing.json")
    try:
        window.set_language(language)
        window.show()
        for width, height, size in ((1100, 760, 14), (1920, 1080, 18), (1100, 760, 14)):
            window.resize(width, height)
            application.processEvents()
            assert window.size().toTuple() == (width, height)
            assert window.apply_camera.font().pixelSize() == size
            assert window.x_value.font().pixelSize() == window.y_value.font().pixelSize() == 2 * size
            assert window.trend_plot.getAxis("bottom").style["tickFont"].pixelSize() == size
            assert not window.live_pane.plot.getAxis("bottom").isVisible()
            for button in (window.tools_button, window.motion_controls.stop_button,
                           window.motion_controls.align_button):
                assert not button.visibleRegion().isEmpty()
                assert button.width() >= button.sizeHint().width()
        window.showFullScreen()
        application.processEvents()
        assert window.isFullScreen()
        assert window.x_value.font().pixelSize() == 2 * window.apply_camera.font().pixelSize()
        window.showNormal()
        application.processEvents()
        assert window.apply_camera.font().pixelSize() == 14
        window.dock.show()
        window.dock.resize(1920, 1080)
        application.processEvents()
        assert window.dock.capture.font().pixelSize() == 18
        assert window.apply_camera.font().pixelSize() == 14
    finally:
        window.close()


@pytest.mark.parametrize("save_position", [False, True])
def test_crosshair_save_is_explicit_and_restores_locked(application, tmp_path, save_position):
    path = tmp_path / "settings.json"
    window = MainWindow(settings_path=path)
    try:
        window.crosshair_toggle.setChecked(True)
        window.move_crosshair_toggle.setChecked(True)
        window._set_crosshair_position((123.5, 45.5))
        if save_position:
            window.save_crosshair_button.click()
            assert not window.move_crosshair_toggle.isChecked()
            assert json.loads(path.read_text())["ui"]["crosshair_position"] == [123.5, 45.5]
        window.move_crosshair_toggle.setChecked(True)
        window._set_crosshair_position((234.5, 67.5))
        # Unrelated autosaves and closing must not commit an unsaved move.
        window.tolerance.setValue(3)
        window._save_startup_settings()
    finally:
        window.close()

    restored = MainWindow(settings_path=path)
    try:
        assert restored.crosshair_toggle.isChecked()
        assert not restored.move_crosshair_toggle.isChecked()
        expected = (123.5, 45.5) if save_position else None
        assert restored.reference_pane.crosshair_position == expected
        assert restored.live_pane.crosshair_position == expected
        restored.center_crosshair_button.click()
        restored.save_crosshair_button.click()
        assert json.loads(path.read_text())["ui"]["crosshair_position"] is None
    finally:
        restored.close()


def test_crosshair_saved_position_roundtrips_in_settings_profile(application, tmp_path, monkeypatch):
    path = tmp_path / "profile.json"
    window = MainWindow(settings_path=tmp_path / "startup.json")
    try:
        window.crosshair_toggle.setChecked(True)
        window._set_crosshair_position((123.5, 45.5))
        window.save_crosshair_button.click()
        monkeypatch.setattr(window, "_save_path", lambda *args: str(path))
        window.save_settings()
        window.center_crosshair_button.click()
        window.save_crosshair_button.click()
        window.crosshair_toggle.setChecked(False)
        monkeypatch.setattr(window, "_choose_file", lambda *args: str(path))
        window.load_settings()
        assert window.crosshair_toggle.isChecked()
        assert window.live_pane.crosshair_position == (123.5, 45.5)
        assert window.reference_pane.crosshair_position == (123.5, 45.5)
        assert window._saved_crosshair_position == (123.5, 45.5)
        assert not window.move_crosshair_toggle.isChecked()
    finally:
        window.close()


def test_corrupt_settings_do_not_prevent_startup(application, tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("broken json")
    window = MainWindow(settings_path=path)
    try:
        assert "Einstellungen konnten nicht wiederhergestellt werden" in window.notice.text()
        assert window.motion_dock.config.pulse_ms == 80
    finally:
        window.close()
    assert load_profile(path, include_motion=True)[4]["pulse_ms"] == 80


def test_unwritable_preferences_do_not_prevent_closing(application, tmp_path):
    blocked = tmp_path / "file-instead-of-directory"
    blocked.write_text("keep")
    window = MainWindow(settings_path=blocked / "settings.json")
    window.motion_dock.frequency.setValue(5)
    assert window.close()
    assert not window.motion.is_alive()
    assert "Starteinstellungen konnten nicht gespeichert werden" in window.statusBar().currentMessage()
    assert blocked.read_text() == "keep"
