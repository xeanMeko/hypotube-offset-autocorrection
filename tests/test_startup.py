"""Exercise GUI imports in fresh processes, before another test initializes Qt."""
import os
import json
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize(
    ("module_name", "widget_name"),
    [
        ("measure_shift.app", "MainWindow"),
        ("measure_shift.widgets", "ImagePane"),
    ],
)
def test_fresh_gui_startup(module_name, widget_name):
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = environment.get("MEASURE_SHIFT_TEST_QPA_PLATFORM") or "offscreen"
    script = textwrap.dedent(
        """
        import importlib
        import sys

        # Import the application module before Qt, as the console launcher does.
        module = importlib.import_module(sys.argv[1])
        from PySide6 import QtWidgets

        application = QtWidgets.QApplication([])
        widget_class = getattr(module, sys.argv[2])

        class UnavailableHardware:
            def connect(self, *args):
                raise RuntimeError("Motion hardware is unavailable in tests")

            def shutdown(self, timeout_s=2):
                pass

        widget = (widget_class("Startup") if sys.argv[2] == "ImagePane"
                  else widget_class(motion_hardware=UnavailableHardware()))
        widget.show()
        application.processEvents()
        widget.close()
        application.processEvents()
        """
    )
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-u", "-c", script, module_name, widget_name],
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, (
        f"Fresh {module_name} startup exited with {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.parametrize("arguments,expected", [
    ([], ("basler", "remembered-camera")),
    (["--demo"], ("demo", "")),
    (["--source", "ids", "--device", "override-camera"], ("ids", "override-camera")),
])
def test_launcher_remembers_source_and_honors_command_line_overrides(arguments, expected):
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    script = textwrap.dedent(
        """
        import json
        import sys
        from measure_shift import app
        from measure_shift.models import MatchConfig
        from measure_shift.storage import save_profile

        path = app.default_settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        save_profile(path, MatchConfig(), 1, 1,
                     ui={"source_kind": "Basler", "device_id": "remembered-camera"})

        class Window(app.MainWindow):
            def __init__(self):
                class UnavailableHardware:
                    def connect(self, *args):
                        raise RuntimeError("Motion hardware is unavailable in tests")

                    def shutdown(self, timeout_s=2):
                        pass

                super().__init__(motion_hardware=UnavailableHardware())

            def start_source(self, kind, **kwargs):
                print(json.dumps([kind, kwargs["identifier"]]), flush=True)
                self.close()
                app.QtCore.QTimer.singleShot(0, app.QtWidgets.QApplication.instance().quit)

        app.MainWindow = Window
        sys.exit(app.main(sys.argv[1:]))
        """
    )
    result = subprocess.run([sys.executable, "-c", script, *arguments], env=environment,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip()) == list(expected)
