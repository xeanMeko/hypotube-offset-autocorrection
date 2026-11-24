"""Keep automatic desktop preferences isolated from real user settings."""
import os
from pathlib import Path

import pytest


if os.name == "nt":
    # Qt's offscreen plugin on Windows needs a font directory for layout tests.
    os.environ.setdefault("QT_QPA_FONTDIR", str(Path(os.environ["WINDIR"]) / "Fonts"))


@pytest.fixture(autouse=True)
def isolated_startup_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    if os.name == "nt":
        # Windows resolves this through its known-folders API instead of XDG.
        from PySide6.QtCore import QStandardPaths
        original = QStandardPaths.writableLocation
        location = QStandardPaths.StandardLocation.GenericConfigLocation
        monkeypatch.setattr(QStandardPaths, "writableLocation",
                            lambda kind: str(tmp_path / "config") if kind == location else original(kind))


@pytest.fixture(autouse=True)
def isolated_default_motion_hardware(monkeypatch):
    class UnavailableHardware:
        def connect(self, *args):
            raise RuntimeError("Motion hardware is unavailable in tests")

        def shutdown(self, timeout_s=2):
            pass

    monkeypatch.setattr("measure_shift.motion.TinkerforgeMotionHardware", UnavailableHardware)
    monkeypatch.setattr("measure_shift.app.master_brick_access", lambda host: [])
