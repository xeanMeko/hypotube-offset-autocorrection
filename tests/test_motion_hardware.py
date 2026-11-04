"""Tinkerforge adapter contracts without SDK installation or physical hardware."""
import importlib
from types import SimpleNamespace

import pytest

from measure_shift.motion_hardware import MotionHardwareError, TinkerforgeMotionHardware
from measure_shift.motion import MotionController


@pytest.fixture
def sdk():
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
            identities = fake.identities or {"2eoQ": 2166, "27s4": 2100 if fake.version == 2 else 223}
            for uid, identifier in identities.items():
                if uid not in fake.missing:
                    self.callbacks[self.CALLBACK_ENUMERATE](uid, "master", "a", [2, 0, 0], [2, 0, 0], identifier, 0)

        def disconnect(self):
            fake.calls.append(("disconnect",))

    class SilentStepper:
        DEVICE_IDENTIFIER = 2166
        OVER_TEMPERATURE_LIMIT = 2

        def __init__(self, uid, ipcon):
            fake.stepper = self
            self.position = 80
            self.velocity = getattr(fake, "initial_velocity", 0)
            self.remaining = getattr(fake, "initial_remaining", 0)
            self.enabled = getattr(fake, "initial_enabled", False)
            self.current = 640
            self.resolution = 5
            self.interpolation = True
            self.input_voltage = 24000
            self.short_to_ground = 0
            self.over_temperature = 0
            self.motor_stalled = False
            self.stop_velocities = getattr(fake, "initial_stop_velocities", None)
            self.fail_stop = getattr(fake, "initial_fail_stop", False)

        def set_response_expected_all(self, enabled):
            fake.calls.append(("stepper_responses", enabled))

        def get_all_data(self):
            return SimpleNamespace(current_position=self.position, current_velocity=self.velocity,
                                   remaining_steps=self.remaining, input_voltage=self.input_voltage,
                                   current_consumption=100)

        def get_current_velocity(self):
            if self.stop_velocities:
                self.velocity = self.stop_velocities.pop(0)
            fake.calls.append(("velocity", self.velocity))
            return self.velocity

        def get_step_configuration(self):
            return self.resolution, self.interpolation

        def get_driver_status(self):
            return SimpleNamespace(open_load=0, short_to_ground=self.short_to_ground,
                                   over_temperature=self.over_temperature, motor_stalled=self.motor_stalled,
                                   actual_motor_current=16, full_step_active=False,
                                   stallguard_result=100, stealth_voltage_amplitude=50)

        def get_enabled(self):
            return self.enabled

        def get_motor_current(self):
            return self.current

        def set_motor_current(self, current):
            fake.calls.append(("current", current))
            self.current = current

        def set_step_configuration(self, resolution, interpolation):
            fake.calls.append(("resolution", resolution, interpolation))
            self.resolution, self.interpolation = resolution, interpolation

        def set_time_base(self, time_base):
            fake.calls.append(("time_base", time_base))

        def set_speed_ramping(self, acceleration, deceleration):
            fake.calls.append(("ramping", acceleration, deceleration))

        def set_enabled(self, enabled):
            fake.calls.append(("enabled", enabled))
            self.enabled = enabled

        def set_max_velocity(self, velocity):
            fake.calls.append(("max_velocity", velocity))
            self.max_velocity = velocity

        def set_steps(self, steps):
            fake.calls.append(("steps", steps))
            self.remaining = steps
            self.velocity = self.max_velocity
            if abs(steps) == 1 and getattr(fake, "single_step_quirk", False):
                # Firmware makes the step without updating its step counter
                # or internal STEPPER_STATE_STEPS state.
                self.position = (self.position + steps + 2 ** 31) % (2 ** 32) - 2 ** 31
                self.velocity = 0
                self.remaining = getattr(fake, "single_step_remaining", steps)

        def stop(self):
            fake.calls.append(("stop",))
            if not self.enabled:
                raise RuntimeError("Got invalid parameter for function 18 (-9)")
            if self.fail_stop:
                raise TimeoutError("transport timeout")
            self.remaining = 0
            if self.stop_velocities is None:
                self.velocity = 0

        def full_brake(self):
            fake.calls.append(("full_brake",))
            self.velocity = self.remaining = 0

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

    modules = {
        "tinkerforge.ip_connection": SimpleNamespace(IPConnection=IPConnection),
        "tinkerforge.bricklet_silent_stepper_v2": SimpleNamespace(BrickletSilentStepperV2=SilentStepper),
        "tinkerforge.bricklet_industrial_digital_in_4": SimpleNamespace(BrickletIndustrialDigitalIn4=DigitalV1),
        "tinkerforge.bricklet_industrial_digital_in_4_v2": SimpleNamespace(BrickletIndustrialDigitalIn4V2=DigitalV2),
    }

    def load(name):
        fake.loaded.append(name)
        return modules[name]

    fake.adapter = TinkerforgeMotionHardware(load, enumeration_timeout_s=.001)
    return fake


def connect(sdk):
    sdk.adapter.connect("localhost", 4223, "2eoQ", "27s4")
    return sdk.adapter


def test_sdk_is_optional_and_not_imported_until_connect(sdk):
    assert sdk.loaded == []
    assert not sdk.adapter.connected

    def missing(name):
        raise ImportError("not installed")

    adapter = TinkerforgeMotionHardware(missing)
    with pytest.raises(MotionHardwareError, match=r"measure-shift\[motion\]"):
        adapter.connect("localhost", 4223, "2eoQ", "27s4")


@pytest.mark.parametrize("version", [1, 2])
def test_identity_detection_normalizes_inputs_without_motor_writes(sdk, version):
    sdk.version = version
    adapter = connect(sdk)
    assert adapter.connected
    assert adapter.read_inputs() == (False, True, False, True)
    assert not any(call[0] in {"current", "resolution", "enabled", "steps"} for call in sdk.calls)
    assert ("auto_reconnect", False) in sdk.calls
    assert ("stepper_responses", True) in sdk.calls
    assert ("digital_responses", True) in sdk.calls
    state = adapter.read_state()
    assert (state.position, state.velocity, state.remaining, state.step_resolution) == (80, 0, 0, 8)
    assert state.motor_current_ma == 640 and not state.enabled and state.fault is None


@pytest.mark.parametrize("identities, message", [
    ({"2eoQ": 999, "27s4": 2100}, "not a Silent Stepper"),
    ({"2eoQ": 2166, "27s4": 999}, "not an Industrial Digital"),
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


def test_connect_disables_previously_enabled_idle_motor_without_changing_setup(sdk):
    sdk.initial_enabled = True
    adapter = connect(sdk)
    assert adapter.connected and not sdk.stepper.enabled
    meaningful = [call for call in sdk.calls if call[0] in {"stop", "velocity", "enabled", "disconnect"}]
    assert meaningful == [("stop",), ("velocity", 0), ("enabled", False)]
    assert not any(call[0] in {"current", "resolution", "steps"} for call in sdk.calls)


def test_connect_stops_previously_moving_motor_before_disabling(sdk):
    sdk.initial_enabled = True
    sdk.initial_velocity = 100
    sdk.initial_remaining = 50
    sdk.initial_stop_velocities = [100, 10, 0]
    adapter = connect(sdk)
    assert adapter.connected and not sdk.stepper.enabled
    meaningful = [call for call in sdk.calls if call[0] in {"stop", "velocity", "enabled", "disconnect"}]
    assert meaningful == [("stop",), ("velocity", 100), ("velocity", 10), ("velocity", 0), ("enabled", False)]
    assert not any(call[0] in {"current", "resolution", "steps"} for call in sdk.calls)


def test_connect_stop_failure_never_disables_possibly_moving_motor(sdk):
    sdk.initial_enabled = True
    sdk.initial_velocity = 100
    sdk.initial_fail_stop = True
    with pytest.raises(MotionHardwareError, match="Cannot stop motor during connection"):
        connect(sdk)
    assert ("enabled", False) not in sdk.calls
    assert sdk.calls[-1] == ("disconnect",)
    assert not sdk.adapter.connected


def test_connect_stop_timeout_does_not_remove_motor_power(sdk, monkeypatch):
    sdk.initial_enabled = True
    sdk.initial_velocity = 100
    sdk.initial_stop_velocities = [100] * 1000
    original = sdk.adapter._stop_and_disable

    def short_stop(timeout_s, *, context):
        assert timeout_s == 2.0
        return original(.015, context=context)

    monkeypatch.setattr(sdk.adapter, "_stop_and_disable", short_stop)
    with pytest.raises(MotionHardwareError, match="before connection timeout"):
        connect(sdk)
    assert ("enabled", False) not in sdk.calls
    assert sdk.calls[-1] == ("disconnect",)


def test_configure_and_move_use_microsteps_finite_moves_and_no_ramping(sdk):
    adapter = connect(sdk)
    adapter.configure(500, 16)
    assert ("current", 500) in sdk.calls
    assert ("resolution", 4, True) in sdk.calls
    assert adapter.read_state().step_resolution == 16
    adapter.enable()
    adapter.move_steps(-37, 80)
    assert sdk.calls[-4:] == [("time_base", 1), ("ramping", 0, 0), ("max_velocity", 80), ("steps", -37)]
    with pytest.raises(MotionHardwareError, match="Wait for the motor"):
        adapter.move_steps(1, 80)
    with pytest.raises(MotionHardwareError, match="Wait for the motor"):
        adapter.configure(500, 8)
    with pytest.raises(MotionHardwareError, match="Wait for the motor"):
        adapter.disable()
    adapter.stop()
    adapter.disable()
    assert not sdk.stepper.enabled


@pytest.mark.parametrize("current,resolution", [(0, 8), (2000, 8), (500, 3), (500, 512)])
def test_invalid_motor_settings_never_reach_device(sdk, current, resolution):
    adapter = connect(sdk)
    with pytest.raises(MotionHardwareError):
        adapter.configure(current, resolution)
    assert not any(call[0] == "current" for call in sdk.calls)


@pytest.mark.parametrize("steps,velocity", [(0, 20), (2**31, 20), (1, 0), (1, 65536), (1.2, 20)])
def test_invalid_move_never_reaches_device(sdk, steps, velocity):
    adapter = connect(sdk)
    with pytest.raises(MotionHardwareError):
        adapter.move_steps(steps, velocity)
    assert not any(call[0] == "steps" for call in sdk.calls)


def test_move_requires_enabled_driver(sdk):
    adapter = connect(sdk)
    with pytest.raises(MotionHardwareError, match="Enable motor power"):
        adapter.move_steps(10, 20)


@pytest.mark.parametrize("steps", [-1, 1])
@pytest.mark.parametrize("remaining", [None, 37, -37])
def test_single_step_firmware_quirk_is_verified_and_cleared_before_repeating(sdk, steps, remaining):
    adapter = connect(sdk)
    adapter.enable()
    sdk.single_step_quirk = True
    if remaining is not None:
        sdk.single_step_remaining = remaining
    start = sdk.stepper.position
    sdk.calls.clear()
    for index in range(1, 4):
        adapter.move_steps(steps, 200)
        assert sdk.stepper.remaining != 0
        state = adapter.read_state()
        assert state.position == start + index * steps
        assert state.velocity == state.remaining == 0
        assert state.enabled
    assert sdk.calls.count(("steps", steps)) == 3
    assert sdk.calls.count(("stop",)) == 3
    adapter.read_state()
    assert sdk.calls.count(("stop",)) == 3
    assert not any(call == ("enabled", False) for call in sdk.calls)


@pytest.mark.parametrize("position,steps", [(2 ** 31 - 1, 1), (-2 ** 31, -1)])
def test_single_step_completion_handles_signed_position_wrap(sdk, position, steps):
    adapter = connect(sdk)
    adapter.enable()
    sdk.single_step_quirk = True
    sdk.stepper.position = position
    adapter.move_steps(steps, 200)
    state = adapter.read_state()
    assert state.position == (-2 ** 31 if steps == 1 else 2 ** 31 - 1)
    assert state.velocity == state.remaining == 0


@pytest.mark.parametrize("steps,position,velocity,remaining", [
    (1, 80, 0, 1),  # The requested step has not happened.
    (1, 81, 200, 1),  # Still moving, even though the target was observed.
    (2, 82, 0, 2),  # The workaround must only apply to single steps.
    (1, 81, 0, 0),  # A firmware version that already reports completion.
])
def test_single_step_workaround_does_not_stop_other_states(sdk, steps, position, velocity, remaining):
    adapter = connect(sdk)
    adapter.enable()
    adapter.move_steps(steps, 200)
    sdk.stepper.position, sdk.stepper.velocity, sdk.stepper.remaining = position, velocity, remaining
    sdk.calls.clear()
    state = adapter.read_state()
    assert (state.position, state.velocity, state.remaining) == (position, velocity, remaining)
    assert ("stop",) not in sdk.calls


def test_single_step_cleanup_failure_does_not_claim_completion(sdk):
    adapter = connect(sdk)
    adapter.enable()
    sdk.single_step_quirk = True
    adapter.move_steps(1, 200)
    sdk.stepper.fail_stop = True
    with pytest.raises(MotionHardwareError, match="Cannot finalize completed single step"):
        adapter.read_state()
    assert sdk.stepper.remaining == 1
    assert sdk.stepper.enabled


def test_single_step_cleanup_requires_fresh_stop_confirmation(sdk, monkeypatch):
    adapter = connect(sdk)
    adapter.enable()
    sdk.single_step_quirk = True
    adapter.move_steps(1, 200)
    monkeypatch.setattr(sdk.stepper, "stop", lambda: None)
    assert adapter.read_state().remaining == 1


@pytest.mark.parametrize("channel,steps", [(2, 1), (1, -1)])
def test_joystick_single_steps_repeat_and_release_without_timeout(sdk, channel, steps):
    adapter = connect(sdk)
    adapter.configure(800)
    sdk.single_step_quirk = True
    sdk.digital.value = [False] * 4
    controller = MotionController()
    controller.on_connected(adapter.read_state())
    controller.on_setup_applied()
    controller.arm()

    def tick(now):
        actions = controller.tick(now, adapter.read_state(), adapter.read_inputs())
        for action in actions:
            if action.kind == "enable":
                adapter.enable()
            elif action.kind == "move":
                adapter.enable()
                adapter.move_steps(action.steps, action.velocity)
            else:
                pytest.fail(f"Unexpected joystick action: {action.kind}")

    tick(0)
    start = sdk.stepper.position
    sdk.digital.value[channel] = True
    tick(1)
    for now in (1.04, 1.55, 2.06):
        tick(now)
    assert sdk.stepper.position == start + 3 * steps
    sdk.digital.value[channel] = False
    tick(2.1)
    tick(10)  # Beyond the old move timeout, with no further movement.
    assert controller.armed and controller.phase == "ready"
    assert controller.state.velocity == controller.state.remaining == 0
    assert sdk.stepper.position == start + 3 * steps


@pytest.mark.parametrize("attribute,value,match", [
    ("short_to_ground", 1, "short to ground"),
    ("over_temperature", 2, "overtemperature"),
    ("motor_stalled", True, "stalled"),
    ("input_voltage", 5000, "below 8 V"),
])
def test_driver_faults_block_enable_and_moves(sdk, attribute, value, match):
    adapter = connect(sdk)
    setattr(sdk.stepper, attribute, value)
    assert match in adapter.read_state().fault
    with pytest.raises(MotionHardwareError, match=match):
        adapter.enable()
    sdk.stepper.enabled = True
    with pytest.raises(MotionHardwareError, match=match):
        adapter.move_steps(10, 20)


def test_disconnection_invalidates_adapter_without_automatic_resume(sdk):
    adapter = connect(sdk)
    sdk.connection.callbacks[sdk.connection.CALLBACK_DISCONNECTED](1)
    assert not adapter.connected
    with pytest.raises(MotionHardwareError, match="lost; reconnect"):
        adapter.read_inputs()
    with pytest.raises(MotionHardwareError, match="lost; reconnect"):
        adapter.move_steps(10, 20)


def test_device_removal_uses_only_valid_disconnected_callback_fields(sdk):
    adapter = connect(sdk)
    sdk.connection.callbacks[sdk.connection.CALLBACK_ENUMERATE]("27s4", None, None, None, None, None, 2)
    with pytest.raises(MotionHardwareError, match="27s4 was disconnected"):
        adapter.read_state()


def test_device_replug_invalidates_configuration_and_never_resumes(sdk):
    adapter = connect(sdk)
    sdk.connection.callbacks[sdk.connection.CALLBACK_ENUMERATE]("2eoQ", "master", "a", [2, 0, 0], [2, 0, 0], 2166, 1)
    with pytest.raises(MotionHardwareError, match="replaced or reconnected"):
        adapter.enable()


def test_old_connection_callback_does_not_invalidate_a_new_connection(sdk):
    adapter = connect(sdk)
    old_disconnected = sdk.connection.callbacks[sdk.connection.CALLBACK_DISCONNECTED]
    adapter.shutdown()
    connect(sdk)
    old_disconnected(1)
    assert adapter.connected


def test_shutdown_stops_then_waits_for_zero_before_disabling(sdk):
    adapter = connect(sdk)
    adapter.enable()
    adapter.move_steps(500, 80)
    sdk.stepper.stop_velocities = [80, 10, 0]
    sdk.calls.clear()
    adapter.shutdown()
    meaningful = [call for call in sdk.calls if call[0] != "timeout"]
    assert meaningful == [("stop",), ("velocity", 80), ("velocity", 10), ("velocity", 0), ("enabled", False), ("disconnect",)]
    assert not adapter.connected
    adapter.shutdown()  # Idempotent after successful cleanup.


def test_shutdown_timeout_never_disables_possibly_moving_motor(sdk):
    adapter = connect(sdk)
    sdk.stepper.enabled = True
    sdk.stepper.velocity = 80
    sdk.stepper.stop_velocities = [80] * 100
    with pytest.raises(MotionHardwareError, match="did not confirm stopping"):
        adapter.shutdown(timeout_s=.015)
    assert ("enabled", False) not in sdk.calls
    assert sdk.calls[-1] == ("disconnect",)


def test_shutdown_stop_failure_is_reported_and_never_disables_driver(sdk):
    adapter = connect(sdk)
    adapter.enable()
    sdk.stepper.fail_stop = True
    with pytest.raises(MotionHardwareError, match="transport timeout"):
        adapter.shutdown()
    assert ("enabled", False) not in sdk.calls
    assert sdk.calls[-1] == ("disconnect",)


def test_emergency_stop_uses_full_brake_and_ordinary_stop_does_not(sdk):
    adapter = connect(sdk)
    adapter.enable()
    adapter.stop()
    assert sdk.calls[-1] == ("stop",)
    assert ("full_brake",) not in sdk.calls
    adapter.emergency_stop()
    assert sdk.calls[-1] == ("full_brake",)


def test_stop_and_shutdown_already_disabled_motor_do_not_send_stop_or_power_commands(sdk):
    adapter = connect(sdk)
    sdk.calls.clear()
    adapter.stop()
    adapter.shutdown()
    assert not any(call[0] in {"stop", "full_brake", "enabled"} for call in sdk.calls)
    assert sdk.calls[-1] == ("disconnect",)
    assert not adapter.connected


def test_motor_setup_works_before_first_enable_and_shutdown_after_disable(sdk):
    adapter = connect(sdk)
    # The worker stops and disables the motor before applying setup.
    adapter.stop()
    adapter.disable()
    adapter.configure(500, 8)
    adapter.enable()
    adapter.move_steps(20, 40)
    adapter.stop()
    adapter.disable()
    sdk.calls.clear()
    adapter.stop()
    adapter.shutdown()
    assert not any(call[0] in {"stop", "enabled"} for call in sdk.calls)
    assert sdk.calls[-1] == ("disconnect",)


@pytest.mark.parametrize("velocity,remaining", [(80, 0), (0, 20)])
@pytest.mark.parametrize("operation", ["stop", "shutdown"])
def test_disabled_driver_reporting_motion_is_not_treated_as_stopped(sdk, velocity, remaining, operation):
    adapter = connect(sdk)
    sdk.stepper.velocity = velocity
    sdk.stepper.remaining = remaining
    with pytest.raises(MotionHardwareError, match="power is disabled but motion is still reported"):
        getattr(adapter, operation)()
    assert not any(call[0] in {"stop", "enabled"} for call in sdk.calls)
    if operation == "shutdown":
        assert sdk.calls[-1] == ("disconnect",)


@pytest.mark.parametrize("getter", ["get_enabled", "get_all_data"])
@pytest.mark.parametrize("operation", ["stop", "shutdown"])
def test_cannot_assume_disabled_motor_stopped_when_read_fails(sdk, monkeypatch, getter, operation):
    adapter = connect(sdk)

    def fail_read():
        raise TimeoutError("transport timeout")

    monkeypatch.setattr(sdk.stepper, getter, fail_read)
    with pytest.raises(MotionHardwareError, match="transport timeout"):
        getattr(adapter, operation)()
    assert not any(call[0] in {"stop", "enabled"} for call in sdk.calls)
    if operation == "shutdown":
        assert sdk.calls[-1] == ("disconnect",)


@pytest.mark.parametrize("operation", ["stop", "shutdown"])
def test_stop_rejection_while_enabled_is_still_reported(sdk, monkeypatch, operation):
    adapter = connect(sdk)
    adapter.enable()

    def rejected_stop():
        raise RuntimeError("Got invalid parameter for function 18 (-9)")

    monkeypatch.setattr(sdk.stepper, "stop", rejected_stop)
    with pytest.raises(MotionHardwareError, match="Got invalid parameter for function 18"):
        getattr(adapter, operation)()
    assert sdk.stepper.enabled
    assert ("enabled", False) not in sdk.calls
    if operation == "shutdown":
        assert sdk.calls[-1] == ("disconnect",)


@pytest.mark.parametrize("digital_version", [1, 2])
def test_adapter_uses_real_optional_sdk_bindings_with_memory_transport(monkeypatch, digital_version):
    """Exercise actual SDK validation, method signatures and return namedtuples.

    The SDK is optional, and every transport operation is in memory. The real
    IPConnection.connect is blocked so this test cannot contact any hardware.
    """
    ip_module = pytest.importorskip("tinkerforge.ip_connection")
    ss_module = pytest.importorskip("tinkerforge.bricklet_silent_stepper_v2")
    di1_module = pytest.importorskip("tinkerforge.bricklet_industrial_digital_in_4")
    di2_module = pytest.importorskip("tinkerforge.bricklet_industrial_digital_in_4_v2")
    ss_class = ss_module.BrickletSilentStepperV2
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
            self.enabled = False
            self.current = 640
            self.resolution = ss_class.STEP_RESOLUTION_8
            self.interpolation = True
            self.position = 20
            self.short_to_ground = 0
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
            for uid, cls in (("2eoQ", ss_class), ("27s4", digital_class)):
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
            if function_id == ss_class.FUNCTION_GET_ALL_DATA:
                return 0, self.position, 0, 24000, 100
            if function_id == ss_class.FUNCTION_GET_DRIVER_STATUS:
                # short_to_ground is an integer phase bitmask, not a bool.
                return 0, self.short_to_ground, 0, False, 16, False, 100, 50
            if function_id == ss_class.FUNCTION_GET_STEP_CONFIGURATION:
                return self.resolution, self.interpolation
            if function_id == ss_class.FUNCTION_GET_ENABLED:
                return self.enabled
            if function_id == ss_class.FUNCTION_GET_MOTOR_CURRENT:
                return self.current
            if function_id == ss_class.FUNCTION_GET_CURRENT_VELOCITY:
                return 0
            if function_id == ss_class.FUNCTION_SET_MOTOR_CURRENT:
                self.current, = parameters
            elif function_id == ss_class.FUNCTION_SET_STEP_CONFIGURATION:
                self.resolution, self.interpolation = parameters
            elif function_id == ss_class.FUNCTION_SET_ENABLED:
                self.enabled, = parameters
            elif function_id == ss_class.FUNCTION_SET_STEPS:
                self.position += parameters[0]
            elif function_id == ss_class.FUNCTION_STOP:
                if not self.enabled:
                    raise ip_module.Error(ip_module.Error.INVALID_PARAMETER,
                                          "Got invalid parameter for function 18")
            else:
                assert function_id in {
                    ss_class.FUNCTION_SET_TIME_BASE, ss_class.FUNCTION_SET_SPEED_RAMPING,
                    ss_class.FUNCTION_SET_MAX_VELOCITY, ss_class.FUNCTION_STOP,
                    ss_class.FUNCTION_FULL_BRAKE,
                }
            return None

    def load(name):
        return SimpleNamespace(IPConnection=MemoryConnection) if name == "tinkerforge.ip_connection" else importlib.import_module(name)

    adapter = TinkerforgeMotionHardware(load)
    adapter.connect("memory-only", 4223, "2eoQ", "27s4")
    adapter.stop()  # Disabled STOP would be rejected by the firmware transport.
    assert not any(fid == ss_class.FUNCTION_STOP for uid, fid, params in transport.requests)
    assert adapter.read_inputs() == (False, True, False, True)
    assert adapter.read_state().position == 20
    adapter.configure(500, 32, False)
    state = adapter.read_state()
    assert (state.motor_current_ma, state.step_resolution, state.interpolation) == (500, 32, False)
    adapter.enable()
    adapter.move_steps(-7, 40)
    assert adapter.read_state().position == 13
    assert ("2eoQ", ss_class.FUNCTION_SET_TIME_BASE, (1,)) in transport.requests
    assert ("2eoQ", ss_class.FUNCTION_SET_SPEED_RAMPING, (0, 0)) in transport.requests
    assert ("2eoQ", ss_class.FUNCTION_SET_STEPS, (-7,)) in transport.requests
    assert all(device.get_response_expected(device.FUNCTION_GET_VALUE)
               for device in transport.devices.values() if device.uid_string == "27s4")
    assert transport.devices["2eoQ"].get_response_expected(ss_class.FUNCTION_SET_STEPS)
    transport.short_to_ground = ss_class.SHORT_TO_GROUND_PHASE_B
    assert adapter.read_state().fault == "Motor driver short to ground"
    adapter.stop()
    adapter.emergency_stop()
    adapter.shutdown()
    assert not transport.enabled and transport.disconnected
