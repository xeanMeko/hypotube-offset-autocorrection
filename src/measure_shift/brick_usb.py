"""Detect Linux Master Brick permissions and request one-time desktop authorization."""

import os
from pathlib import Path
import subprocess
import sys


def master_brick_access(host, *, devices=Path("/sys/bus/usb/devices"), bus=Path("/dev/bus/usb")):
    """Report read/write access for each attached Master Brick on a local endpoint."""
    access = []
    if sys.platform != "linux" or host.lower() not in {"localhost", "127.0.0.1", "::1"}:
        return access
    for vendor in devices.glob("*/idVendor"):
        device = vendor.parent
        try:
            if vendor.read_text().strip() != "16d0" or (device / "idProduct").read_text().strip() != "063d":
                continue
            bus_number = int((device / "busnum").read_text())
            address = int((device / "devnum").read_text())
            node = bus / f"{bus_number:03d}" / f"{address:03d}"
            if node.exists():
                access.append(os.access(node, os.R_OK | os.W_OK))
        except (OSError, ValueError):
            continue  # A device can disappear between sysfs reads.
    return access


# pkexec receives only this fixed program. Stage the rule in the root-owned
# target directory so concurrent launches always publish a complete file.
_INSTALL_RULE = """\
set -eu
PATH=/usr/sbin:/usr/bin:/sbin:/bin
export PATH
cd /etc/udev/rules.d
group=''
if getent group plugdev >/dev/null; then group=', GROUP="plugdev"'; fi
tmp=$(mktemp .measure-shift-XXXXXX)
trap 'rm -f -- "$tmp"' EXIT
printf '%s\\n' 'SUBSYSTEM=="usb", ATTR{idVendor}=="16d0", ATTR{idProduct}=="063d", MODE="0660"'"$group"', TAG+="uaccess"' > "$tmp"
chmod 0644 "$tmp"
mv -fT -- "$tmp" 70-measure-shift-master-brick.rules
udevadm control --reload-rules
udevadm trigger --action=add --settle --subsystem-match=usb --attr-match=idVendor=16d0 --attr-match=idProduct=063d
"""


def enable_master_brick_access():
    try:
        result = subprocess.run(
            ["/usr/bin/pkexec", "--disable-internal-agent", "/bin/sh", "-c", _INSTALL_RULE],
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Install pkexec and a desktop polkit authentication agent to enable USB access.") from exc
    if result.returncode == 126:
        raise RuntimeError("USB access setup canceled.")
    if result.returncode:
        raise RuntimeError("Could not enable Master Brick access. " + (
            result.stderr.strip() or "Check administrator authorization and the desktop polkit agent."))
    access = master_brick_access("localhost")
    if not access or False in access:
        raise RuntimeError("USB rule installed. Reconnect the Master Brick, then select Connect hardware.")
