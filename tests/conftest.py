"""Keep automatic desktop preferences isolated from real user settings."""
import pytest


@pytest.fixture(autouse=True)
def isolated_startup_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
