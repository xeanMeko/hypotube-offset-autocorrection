"""Tinkerforge adapter contracts without SDK installation or physical hardware."""
import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from measure_shift.motion_hardware import MotionHardwareError, TinkerforgeMotionHardware
from measure_shift.joystick import Direction, WiredJoystick
from measure_shift.motion import MotionConfig, MotionController, MotionWorker


@pytest.fixture
def sdk(monkeypatch):
    monkeypatch.setattr("measure_shift.motion_hardware.master_brick_access", lambda host: [])
    fake = SimpleNamespace(calls=[], loaded=[], version=2, missing=set(), identities=None)

    class IPConnection:
        CALLBACK_ENUMERATE = 253
        CALLBACK_DISCONNECTED = 1
        ENUMERATION_TYPE_AVAILABLE = 0
        ENUMERATION_TYPE_CONNECTED = 1
        ENUMERATION_TYPE_DISCONNECTED = 2

        def __init__(self):
            fake.connection = self
            self.callbacks = {}

        def set_auto_reconnect(self, enabled):
            fake.calls.append(("auto_reconnect", enabled))

        def set_timeout(self, timeout):
            fake.calls.append(("timeout", timeout))

        def register_callback(self, identifier, callback):
            self.callbacks[identifier] = callback

        def connect(self, host, port):
            fake.calls.append(("connect", host, port))

        def enumerate(self):
            fake.calls.append(("enumerate",))
            identities = fake.identities or {"2eoQ": 2166, "2j7T": 26, "27s4": 2100 if fake.version == 2 else 223}
            for uid, identifier in identities.items():
                if uid not in fake.missing:
                    self.callbacks[self.CALLBACK_ENUMERATE](uid, "master", "a", [2, 0, 0], [2, 0, 0], identifier, 0)

        def disconnect(self):
            fake.calls.append(("disconnect",))

    class DigitalV1:
        DEVICE_IDENTIFIER = 223

        def __init__(self, uid, ipcon):
            fake.digital = self
            self.value = 0b1010
            self.group = getattr(fake, "group", ["n", "n", "n", "n"])

        def set_response_expected_all(self, enabled):
            fake.calls.append(("digital_responses", enabled))

        def get_value(self):
            return self.value

        def get_group(self):
            return self.group

    class DigitalV2(DigitalV1):
        DEVICE_IDENTIFIER = 2100

        def __init__(self, uid, ipcon):
            super().__init__(uid, ipcon)
            self.value = [False, True, False, True]

    class Relay:
        def __init__(self, uid, ipcon):
            assert uid == "2j7T"

        def set_response_expected_all(self, enabled):
            assert enabled

        def set_monoflop(self, coil, state, duration):
            fake.calls.append(("monoflop", coil, state, duration))

    class DualRelay(Relay):
        DEVICE_IDENTIFIER = 26

        def set_state(self, first, second):
            fake.calls.append(("release", first, second))

    class IndustrialDualRelay(Relay):
        DEVICE_IDENTIFIER = 284

        def set_value(self, first, second):
            fake.calls.append(("release", first, second))

    modules = {
        "tinkerforge.ip_connection": SimpleNamespace(IPConnection=IPConnection),
        "tinkerforge.bricklet_dual_relay": SimpleNamespace(BrickletDualRelay=DualRelay),
        "tinkerforge.bricklet_industrial_dual_relay": SimpleNamespace(BrickletIndustrialDualRelay=IndustrialDualRelay),
        "tinkerforge.bricklet_industrial_digital_in_4": SimpleNamespace(BrickletIndustrialDigitalIn4=DigitalV1),
        "tinkerforge.bricklet_industrial_digital_in_4_v2": SimpleNamespace(BrickletIndustrialDigitalIn4V2=DigitalV2),
    }

    def load(name):
        fake.loaded.append(name)
        return modules[name]

    fake.modules = modules
    fake.adapter = TinkerforgeMotionHardware(load, enumeration_timeout_s=.001)
    return fake


def install_bench_stepper(sdk):
    device = Mock(STEP_RESOLUTION_8=5, OVER_TEMPERATURE_LIMIT=2)
    data = SimpleNamespace(current_velocity=0, remaining_steps=0, input_voltage=24000)
    device.get_all_data.return_value = data
    device.get_driver_status.return_value = SimpleNamespace(short_to_ground=0, over_temperature=0, motor_stalled=False)
    device.get_enabled.return_value = False
    device.get_current_velocity.return_value = 0
    device.set_enabled.side_effect = lambda enabled: setattr(device.get_enabled, "return_value", enabled)
    device.set_steps.side_effect = lambda steps: setattr(data, "remaining_steps", steps)
    device.stop.side_effect = lambda: setattr(data, "remaining_steps", 0)
    sdk.modules["tinkerforge.bricklet_silent_stepper_v2"] = SimpleNamespace(
        BrickletSilentStepperV2=Mock(DEVICE_IDENTIFIER=2166, return_value=device))
    return device


def connect(sdk):
    sdk.adapter.connect("localhost", 4223, "27s4", "2j7T")
    return sdk.adapter


def test_sdk_is_optional_and_not_imported_until_connect(sdk):
    assert sdk.loaded == []
    assert not sdk.adapter.connected

    def missing(name):
        raise ImportError("not installed")

    adapter = TinkerforgeMotionHardware(missing)
    with pytest.raises(MotionHardwareError, match=r"measure-shift\[motion\]"):
        adapter.connect("localhost", 4223, "27s4", "2j7T")


@pytest.mark.parametrize("version", [1, 2])
def test_identity_detection_normalizes_inputs_without_stepper_binding(sdk, version):
    sdk.version = version
    adapter = connect(sdk)
    assert adapter.connected
    assert adapter.read_inputs() == (False, True, False, True)
    assert all("stepper" not in module for module in sdk.loaded)
    assert ("auto_reconnect", False) in sdk.calls
    assert ("digital_responses", True) in sdk.calls
    assert ("release", False, False) in sdk.calls


@pytest.mark.parametrize("identities, message", [
    ({"2j7T": 26, "27s4": 999}, "not an Industrial Digital"),
])
def test_wrong_identity_disconnects_without_enabling(sdk, identities, message):
    sdk.identities = identities
    with pytest.raises(MotionHardwareError, match=message):
        connect(sdk)
    assert ("disconnect",) in sdk.calls
    assert not sdk.adapter.connected


def test_missing_uid_reports_actual_uid_and_releases_connection(sdk):
    sdk.missing.add("27s4")
    with pytest.raises(MotionHardwareError, match="not found: 27s4"):
        connect(sdk)
    assert sdk.calls[-1] == ("disconnect",)


def test_bench_mode_requires_the_selected_stepper_before_accepting_presses(sdk):
    motor = install_bench_stepper(sdk)
    sdk.missing.add("2eoQ")
    port = WiredJoystick(sdk.adapter, lambda: MotionConfig(axis_mode="bench"))
    with pytest.raises(MotionHardwareError, match="not found: 2eoQ"):
        port.connect()
    assert not sdk.adapter.connected
    assert not any(call[0] == "monoflop" for call in sdk.calls)
    motor.set_steps.assert_not_called()


@pytest.mark.parametrize("group", [["a", "n", "n", "n"], ["a", "b", "n", "n"], ["n", "a", "n", "n"]])
def test_grouped_v1_is_rejected_without_changing_hardware_group(sdk, group):
    sdk.version = 1
    sdk.group = group.copy()
    with pytest.raises(MotionHardwareError, match="Use Brick Viewer to restore its standalone group"):
        connect(sdk)
    assert sdk.digital.group == group
    assert not sdk.adapter.connected
    assert ("enabled", True) not in sdk.calls
    assert sdk.calls[-1] == ("disconnect",)


def test_v1_runtime_regrouping_is_detected_before_reading_contacts(sdk):
    sdk.version = 1
    adapter = connect(sdk)
    sdk.digital.group = ["a", "b", "n", "n"]
    with pytest.raises(MotionHardwareError, match="channels 0–3 may belong to another Bricklet"):
        adapter.read_inputs()


def test_disconnection_invalidates_adapter_without_automatic_resume(sdk):
    adapter = connect(sdk)
    sdk.connection.callbacks[sdk.connection.CALLBACK_DISCONNECTED](1)
    assert not adapter.connected
    with pytest.raises(MotionHardwareError, match="lost; reconnect"):
        adapter.read_inputs()
    with pytest.raises(MotionHardwareError, match="lost; reconnect"):
        adapter.press_output(0, 80)


def test_device_removal_uses_only_valid_disconnected_callback_fields(sdk):
    adapter = connect(sdk)
    sdk.connection.callbacks[sdk.connection.CALLBACK_ENUMERATE]("27s4", None, None, None, None, None, 2)
    with pytest.raises(MotionHardwareError, match="27s4 was disconnected"):
        adapter.read_inputs()


def test_device_replug_invalidates_configuration_and_never_resumes(sdk):
    adapter = connect(sdk)
    sdk.connection.callbacks[sdk.connection.CALLBACK_ENUMERATE]("2j7T", "master", "a", [2, 0, 0], [2, 0, 0], 26, 1)
    with pytest.raises(MotionHardwareError, match="replaced or reconnected"):
        adapter.press_output(0, 80)


def test_old_connection_callback_does_not_invalidate_a_new_connection(sdk):
    adapter = connect(sdk)
    old_disconnected = sdk.connection.callbacks[sdk.connection.CALLBACK_DISCONNECTED]
    adapter.shutdown()
    connect(sdk)
    old_disconnected(1)
    assert adapter.connected


@pytest.mark.parametrize("digital_version", [1, 2])
def test_adapter_uses_real_optional_sdk_bindings_with_memory_transport(monkeypatch, digital_version):
    """Exercise actual SDK validation, method signatures and return namedtuples.

    The SDK is optional, and every transport operation is in memory. The real
    IPConnection.connect is blocked so this test cannot contact any hardware.
    """
    ip_module = pytest.importorskip("tinkerforge.ip_connection")
    relay_module = pytest.importorskip("tinkerforge.bricklet_dual_relay")
    di1_module = pytest.importorskip("tinkerforge.bricklet_industrial_digital_in_4")
    di2_module = pytest.importorskip("tinkerforge.bricklet_industrial_digital_in_4_v2")
    relay_class = relay_module.BrickletDualRelay
    digital_class = (di1_module.BrickletIndustrialDigitalIn4 if digital_version == 1
                     else di2_module.BrickletIndustrialDigitalIn4V2)
    transport = None

    def forbidden_connect(*args, **kwargs):
        pytest.fail("A hardware test attempted to open a real network connection")

    monkeypatch.setattr(ip_module.IPConnection, "connect", forbidden_connect)

    class MemoryConnection:
        CALLBACK_ENUMERATE = ip_module.IPConnection.CALLBACK_ENUMERATE
        CALLBACK_DISCONNECTED = ip_module.IPConnection.CALLBACK_DISCONNECTED
        ENUMERATION_TYPE_AVAILABLE = ip_module.IPConnection.ENUMERATION_TYPE_AVAILABLE
        ENUMERATION_TYPE_CONNECTED = ip_module.IPConnection.ENUMERATION_TYPE_CONNECTED
        ENUMERATION_TYPE_DISCONNECTED = ip_module.IPConnection.ENUMERATION_TYPE_DISCONNECTED

        def __init__(self):
            nonlocal transport
            transport = self
            self.callbacks = {}
            self.devices = {}
            self.requests = []
            self.disconnected = False

        def add_device(self, device):
            self.devices[device.uid_string] = device

        def set_auto_reconnect(self, enabled):
            assert enabled is False

        def set_timeout(self, timeout):
            assert 0 < timeout <= .25

        def register_callback(self, identifier, callback):
            self.callbacks[identifier] = callback

        def connect(self, host, port):
            assert (host, port) == ("memory-only", 4223)

        def enumerate(self):
            callback = self.callbacks[self.CALLBACK_ENUMERATE]
            for uid, cls in (("2j7T", relay_class), ("27s4", digital_class)):
                callback(uid, "master", "a", [2, 0, 0], [2, 0, 0], cls.DEVICE_IDENTIFIER,
                         self.ENUMERATION_TYPE_AVAILABLE)

        def disconnect(self):
            self.disconnected = True

        def send_request(self, device, function_id, parameters, parameter_format,
                         response_length, response_format):
            self.requests.append((device.uid_string, function_id, parameters))
            if function_id == 255:
                return device.uid_string, "master", "a", [2, 0, 0], [2, 0, 0], device.device_identifier
            if device.uid_string == "27s4":
                if digital_version == 1 and function_id == digital_class.FUNCTION_GET_GROUP:
                    return ["n", "n", "n", "n"]
                assert function_id == digital_class.FUNCTION_GET_VALUE
                return 0b1010 if digital_version == 1 else (False, True, False, True)
            assert function_id in (relay_class.FUNCTION_SET_STATE, relay_class.FUNCTION_SET_MONOFLOP)

    def load(name):
        assert "stepper" not in name
        if name == "tinkerforge.ip_connection":
            return SimpleNamespace(IPConnection=MemoryConnection)
        return importlib.import_module(name)

    adapter = TinkerforgeMotionHardware(load)
    adapter.connect("memory-only", 4223, "27s4", "2j7T")
    assert adapter.read_inputs() == (False, True, False, True)
    adapter.press_output(1, 80)
    adapter.shutdown()
    assert ("2j7T", relay_class.FUNCTION_SET_MONOFLOP, (2, True, 80)) in transport.requests
    assert transport.requests[-1] == ("2j7T", relay_class.FUNCTION_SET_STATE, (False, False))
    assert transport.disconnected


@pytest.mark.parametrize("invalid", [0, 2, -8, True, None])
def test_wired_joystick_rejects_non_direction_commands(sdk, invalid):
    adapter = connect(sdk)
    controller = MotionController()
    joystick = WiredJoystick(adapter, lambda: controller.config)
    sdk.calls.clear()
    with pytest.raises(ValueError, match="LEFT or RIGHT"):
        joystick.pulse(invalid)
    assert sdk.calls == []


@pytest.mark.parametrize("identifier,first_coil,second_coil", [(26, 1, 2), (284, 0, 1)])
def test_factory_relay_pulses_and_releases_both_contacts_without_motor(sdk, identifier, first_coil, second_coil):
    sdk.identities = {"2j7T": identifier}
    adapter = sdk.adapter
    adapter.connect("localhost", 4223, None, "2j7T")
    assert all("stepper" not in name and "digital_in" not in name for name in sdk.loaded)
    assert adapter.read_inputs() == (False,) * 4
    adapter.press_output(0, 80)
    adapter.release_outputs()
    adapter.press_output(1, 80)
    adapter.shutdown()
    assert [c for c in sdk.calls if c[0] in ("monoflop", "release")] == [
        ("release", False, False), ("monoflop", first_coil, True, 80),
        ("release", False, False), ("monoflop", second_coil, True, 80), ("release", False, False),
    ]
    assert not adapter.connected


@pytest.mark.parametrize("identities,message", [
    ({"2j7T": 2100}, "not a Dual Relay"),
    ({"23GY": 224}, "not found: 2j7T"),
    ({"2j7T": 2124}, "not a Dual Relay"),
])
def test_wrong_or_missing_relay_never_operates_contacts(sdk, identities, message):
    sdk.identities = identities
    with pytest.raises(MotionHardwareError, match=message):
        sdk.adapter.connect("localhost", 4223, None, "2j7T")
    assert not any(call[0] in {"monoflop", "release"} for call in sdk.calls)


@pytest.mark.parametrize("stop_early", [False, True])
def test_contact_bounce_after_coil_release_does_not_look_like_an_operator_press(sdk, stop_early):
    sdk.identities = {"2j7T": 284, "27s4": 2100}
    config = MotionConfig()
    now = 0.
    port = WiredJoystick(sdk.adapter, lambda: config, clock=lambda: now)
    port.connect()
    try:
        sdk.digital.value = [False] * 4
        port.pulse(Direction.RIGHT)
        released_at = config.pulse_ms / 1000
        if stop_early:
            now = released_at = .04
            port.stop()
        now = released_at + .005
        sdk.digital.value[1:3] = [True, True]
        assert port.read_inputs() == (False, False, True, False)
        now = released_at + .04
        assert port.read_inputs() == (False, True, True, False)
    finally:
        port.shutdown()


@pytest.mark.parametrize("input_channel,coil", [(2, 1), (1, 0)])
@pytest.mark.parametrize("active_high", [True, False])
@pytest.mark.parametrize("mode", ["bench", "external"])
def test_wired_joystick_press_repeats_at_configured_rate_and_stops_on_release(
        sdk, input_channel, coil, active_high, mode):
    sdk.identities = {"2j7T": 284, "27s4": 2100}
    motor = None
    if mode == "bench":
        sdk.identities["2eoQ"] = 2166
        motor = install_bench_stepper(sdk)
    config = MotionConfig(active_high=active_high, repeat_hz=1, axis_mode=mode)
    now = 0.
    port = WiredJoystick(sdk.adapter, lambda: config, clock=lambda: now)
    worker = MotionWorker(config, joystick=port)
    worker._command("connect", 1)
    sdk.digital.value = [False, not active_high, not active_high, False]

    def sample(at):
        nonlocal now
        now = at
        for action in worker.controller.tick(now, port.read_state(), port.read_inputs()):
            worker._execute(action)

    try:
        sample(.01)
        sdk.digital.value[input_channel] = active_high
        for t in (.1, .14, .16, .22, .26, .31, .35, 1.13):
            sample(t)
        pulse = ("monoflop", coil, True, config.pulse_ms)
        assert [call for call in sdk.calls if call[0] == "monoflop"] == [pulse]
        if motor is not None:
            steps = 8 if input_channel == 2 else -8
            motor.set_steps.assert_called_once_with(steps)
            assert port.read_state().busy  # The relay has already released.
            motor.get_all_data.return_value.remaining_steps = 0
        sample(1.15)
        sdk.digital.value[input_channel] = not active_high
        for t in (1.16, 1.4, 2.2):
            sample(t)
        assert [call for call in sdk.calls if call[0] == "monoflop"] == [pulse, pulse]
        if motor is not None:
            assert [c.args for c in motor.set_steps.call_args_list] == [(steps,), (steps,)]
            port.stop()
            assert not port.read_state().busy and not motor.get_enabled()
        else:
            assert not any("stepper" in name for name in sdk.loaded)
    finally:
        port.shutdown()


def test_bench_failure_after_relay_press_releases_and_stops_without_retry(sdk):
    motor = install_bench_stepper(sdk)
    motor.set_steps.side_effect = RuntimeError("stepper did not acknowledge move")
    config = MotionConfig(axis_mode="bench")
    worker = MotionWorker(config, hardware=sdk.adapter)
    worker.connect_hardware()
    worker.start()
    try:
        from test_motion import wait_for
        wait_for(lambda: hasattr(sdk, "digital"))
        sdk.digital.value = [False] * 4
        wait_for(lambda: worker.output.read()[0].phase == "ready")
        worker.set_keyboard(False, True)
        wait_for(lambda: "did not acknowledge" in worker.output.read()[0].message)
        assert not worker.output.read()[0].connected
        assert [c for c in sdk.calls if c[0] == "monoflop"] == [("monoflop", 1, True, 80)]
        assert sdk.calls[-2:] == [("release", False, False), ("disconnect",)]
        motor.set_steps.assert_called_once_with(-8)
        assert not motor.get_enabled()
    finally:
        worker.stop_worker()
        worker.join(2)


@pytest.mark.parametrize("operation", ["align", "calibrate"])
def test_camera_operations_wait_for_the_bench_motor_after_relay_release(sdk, operation):
    from test_motion import CONTEXT, feedback

    motor = install_bench_stepper(sdk)
    config = MotionConfig(axis_mode="bench")
    now = 0.
    port = WiredJoystick(sdk.adapter, lambda: config, clock=lambda: now)
    barriers = []

    def fresh(after):
        barriers.append(after)
        return len(barriers)

    worker = MotionWorker(config, joystick=port, freshness_callback=fresh)
    worker.controller.set_context(CONTEXT)
    worker._command("connect", 1)
    sdk.digital.value = [False] * 4

    def tick(at, dx):
        nonlocal now
        now = at
        measurement = feedback(at, dx, at, generation=len(barriers))
        for action in worker.controller.tick(now, port.read_state(), port.read_inputs(), measurement):
            worker._execute(action)

    try:
        tick(.05, 4)
        worker._command(operation, 1)
        tick(.1, 4)
        tick(.4, 4)
        motor.set_steps.assert_called_once_with(-8)
        tick(.9, 0)
        assert len(barriers) == 1
        motor.get_all_data.return_value.remaining_steps = 0
        tick(1., 0)
        assert barriers[-1] == pytest.approx(1.2)
        for at in (1.3, 1.4, 1.5):
            tick(at, 0)
        if operation == "align":
            assert worker.controller.message == "X aligned"
        else:
            assert worker.controller.config.pixels_per_right_press == -4
        motor.set_steps.assert_called_once_with(-8)
        assert [c for c in sdk.calls if c[0] == "monoflop"] == [("monoflop", 1, True, 80)]
    finally:
        port.shutdown()


def test_missing_bricklet_with_inaccessible_master_offers_usb_setup(sdk, monkeypatch):
    from measure_shift.motion_hardware import MasterBrickAccessError

    monkeypatch.setattr("measure_shift.motion_hardware.master_brick_access", lambda host: [False])
    sdk.missing.add("27s4")
    with pytest.raises(MasterBrickAccessError):
        connect(sdk)
    assert sdk.calls[-1] == ("disconnect",)
