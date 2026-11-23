"""Keep automatic desktop preferences isolated from real user settings."""
import pytest


@pytest.fixture(autouse=True)
def isolated_startup_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))


@pytest.fixture(autouse=True)
def isolated_default_motion_hardware(monkeypatch):
    class UnavailableHardware:
        def connect(self, *args):
            raise RuntimeError("Motion hardware is unavailable in tests")

        def shutdown(self, timeout_s=2):
            pass

    monkeypatch.setattr("measure_shift.motion.TinkerforgeMotionHardware", UnavailableHardware)
    monkeypatch.setattr("measure_shift.app.master_brick_access", lambda host: [])
