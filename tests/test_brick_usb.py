import subprocess

import pytest

from measure_shift import brick_usb


def test_usb_probe_uses_current_address_and_filters_product_and_host(tmp_path, monkeypatch):
    devices, bus = tmp_path / "devices", tmp_path / "bus"
    node = bus / "004/017"
    node.parent.mkdir(parents=True)
    node.touch()
    usb = devices / "4-3"
    usb.mkdir(parents=True)
    for name, value in dict(idVendor="16d0", idProduct="063d", busnum="4", devnum="17").items():
        (usb / name).write_text(value)
    monkeypatch.setattr(brick_usb.sys, "platform", "linux")
    def probe(host):
        return brick_usb.master_brick_access(host, devices=devices, bus=bus)

    assert probe("localhost") == [True]
    monkeypatch.setattr(brick_usb.os, "access", lambda path, flags: False)
    assert probe("localhost") == [False]
    assert probe("remote") == []
    (usb / "idProduct").write_text("1234")
    assert probe("localhost") == []


@pytest.mark.parametrize("code,access,error", [
    (0, [True], None), (0, [False], "Reconnect"), (0, [], "Reconnect"),
    (126, [False], "canceled"), (127, [False], "authorization"),
])
def test_install_verifies_access(code, access, error, monkeypatch):
    def run(argv, **kwargs):
        assert argv[0] == "/usr/bin/pkexec"
        return subprocess.CompletedProcess(argv, code, "", "")

    monkeypatch.setattr(brick_usb.subprocess, "run", run)
    monkeypatch.setattr(brick_usb, "master_brick_access", lambda host: access)
    if error:
        with pytest.raises(RuntimeError, match=error):
            brick_usb.enable_master_brick_access()
    else:
        brick_usb.enable_master_brick_access()
