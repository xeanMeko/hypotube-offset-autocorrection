"""Launch the Linux bundle's Brick Daemon for the lifetime of the app."""

from contextlib import contextmanager
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

from .motion import MotionConfig


_LOG = logging.getLogger(__name__)
_ENDPOINT = (MotionConfig.host, MotionConfig.port)


def _listening():
    try:
        with socket.create_connection(_ENDPOINT, timeout=0.1):
            return True
    except OSError:
        return False


@contextmanager
def bundled_brickd():
    if sys.platform != "linux" or not getattr(sys, "frozen", False):
        yield
        return

    if _listening():
        yield
        return

    directory = Path(sys.executable).resolve().parent / "brickd"
    environment = os.environ.copy()
    original_library_path = environment.pop("LD_LIBRARY_PATH_ORIG", None)
    if original_library_path is None:
        environment.pop("LD_LIBRARY_PATH", None)
    else:
        environment["LD_LIBRARY_PATH"] = original_library_path

    with TemporaryDirectory(prefix="measure-shift-brickd-") as runtime:
        try:
            process = subprocess.Popen([
                directory / "brickd",
                "--config-file", directory / "brickd.conf",
                "--pid-file", Path(runtime) / "brickd.pid",
            ], env=environment)
        except OSError as exc:
            _LOG.warning("Could not start bundled Brick Daemon: %s", exc)
            yield
            return

        try:
            # The axis attempts its first connection as soon as the window opens.
            deadline = time.monotonic() + 3
            while process.poll() is None and time.monotonic() < deadline and not _listening():
                time.sleep(0.05)
            yield
        finally:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
