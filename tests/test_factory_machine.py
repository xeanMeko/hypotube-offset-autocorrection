"""Offline coverage for the opt-in two-Master bench command."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import factory_machine as rig
from measure_shift.bench_axis import SilentStepperBench
from measure_shift.joystick import Direction, JoystickState
from measure_shift.motion_hardware import MotionHardwareError


@pytest.fixture
def clock(monkeypatch):
    class Clock:
        now = 0.

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    clock = Clock()
    monkeypatch.setattr(rig, "time", clock)
    return clock


@pytest.mark.parametrize("axis,steps,deltas", [("x", 8, [-8, 8, 0, 0]), ("rotation", 32, [0, 0, -32, 32])])
def test_four_contacts_one_physical_axis(clock, capsys, axis, steps, deltas):
    device = Mock()
    device.position = 2**31 - 1
    device.get_current_position.side_effect = lambda: device.position
    device.get_all_data.return_value = SimpleNamespace(current_velocity=0, remaining_steps=0, input_voltage=24000)
    device.get_driver_status.return_value = SimpleNamespace(short_to_ground=False, over_temperature=0, motor_stalled=False)
    bench = SilentStepperBench(device, steps_per_nudge=steps)

    def drive(steps):
        device.position = (device.position + steps + 2**31) % 2**32 - 2**31

    device.set_steps.side_effect = drive
    for channel, delta in enumerate(deltas):
        start = clock.now
        rig.receive_press(lambda: channel if clock.now < start + .08 else None,
                          bench, device, axis, expected=channel)
        assert f"driver delta {delta:+d}" in capsys.readouterr().out
    assert [call.args[0] for call in device.set_steps.call_args_list] == [-steps, steps]


@pytest.mark.parametrize("inputs,match", [
    (lambda t: None, "no closure"),
    (lambda t: 3, "observed IN 3"),
    (lambda t: 0, "to release"),
    (lambda t: 0 if t < .03 else 2, "Unexpected IN 2"),
])
def test_bad_wiring_or_held_contact(clock, inputs, match):
    bench = Mock(STEPS_PER_NUDGE=8)
    with pytest.raises(MotionHardwareError, match=match):
        rig.receive_press(lambda: inputs(clock.now), bench, Mock(), "x", expected=0)
    assert bench.nudge.call_count == (1 if match in ("to release", "Unexpected IN 2") else 0)


def test_stepper_counter_must_confirm_move(clock):
    bench = Mock(STEPS_PER_NUDGE=8)
    bench.read_state.return_value = JoystickState()
    stepper = Mock()
    stepper.get_current_position.return_value = 11
    with pytest.raises(MotionHardwareError, match="expected driver delta -8, observed 0"):
        rig.receive_press(lambda: 0 if clock.now < .08 else None, bench, stepper, "x", expected=0)
    bench.nudge.assert_called_once_with(Direction.RIGHT)


def test_grouped_and_simultaneous_inputs_rejected():
    digital = Mock()
    digital.get_group.return_value = ("a", "n", "n", "n")
    with pytest.raises(MotionHardwareError, match="standalone group"):
        rig.active_channel(digital)
    digital.get_group.return_value = ("n",) * 4
    digital.get_value.return_value = 0b0110
    with pytest.raises(MotionHardwareError, match="Multiple simulator inputs"):
        rig.active_channel(digital)
    digital.get_value.return_value = 0b1000
    assert rig.active_channel(digital) == 3


def test_verify_parent_before_motor_use():
    model = Mock(DEVICE_IDENTIFIER=2166)
    model.return_value.get_identity.return_value = SimpleNamespace(uid=rig.MOTOR, connected_uid=rig.SENDER,
                                                                  device_identifier=2166)
    with pytest.raises(MotionHardwareError, match="Expected 2eke on 6k2K8q"):
        rig.identify(rig.MOTOR, rig.RECEIVER, Mock(), model)
    model.return_value.set_enabled.assert_not_called()


@pytest.mark.parametrize("mode,broken", [("check", False), ("test", False), ("test", True)])
def test_two_master_command_and_cleanup(monkeypatch, clock, mode, broken):
    sdk = pytest.importorskip("tinkerforge.ip_connection")
    connection = Mock()
    monkeypatch.setattr(sdk, "IPConnection", lambda: connection)
    sender = Mock()
    devices = {uid: Mock() for uid in (rig.RELAY, rig.DIGITAL, rig.MOTOR, rig.ROTATION_MOTOR)}
    digital, motor = devices[rig.DIGITAL], devices[rig.MOTOR]
    rotary = devices[rig.ROTATION_MOTOR]
    monkeypatch.setattr(rig, "identify", lambda uid, master, conn, *models: devices[uid])
    monkeypatch.setattr(rig, "TinkerforgeMotionHardware", lambda: sender)

    pulse = [None, 0.]

    def send(channel, ms):
        pulse[:] = [channel, clock.now + ms / 1000]

    sender.press_output.side_effect = send
    digital.get_group.return_value = ("n",) * 4
    digital.get_value.side_effect = lambda: [
        not broken and clock.now < pulse[1] and channel == pulse[0] for channel in range(4)]
    motor.get_enabled.return_value = False
    motor.get_current_velocity.return_value = 0
    motor.get_motor_current.return_value = 800
    motor.get_all_data.return_value = SimpleNamespace(current_velocity=0, remaining_steps=0, input_voltage=24000)
    motor.get_driver_status.return_value = SimpleNamespace(short_to_ground=False, over_temperature=0, motor_stalled=False)
    motor.get_current_position.return_value = 0
    motor.set_steps.side_effect = lambda steps: setattr(motor.get_current_position, "return_value",
                                                        motor.get_current_position.return_value + steps)
    motor.set_enabled.side_effect = lambda enabled: setattr(motor.get_enabled, "return_value", enabled)
    rotary.get_enabled.return_value = False
    rotary.get_current_velocity.return_value = 0
    rotary.get_motor_current.return_value = 800
    rotary.get_all_data.return_value = motor.get_all_data.return_value
    rotary.get_driver_status.return_value = motor.get_driver_status.return_value
    rotary.get_current_position.return_value = 0
    rotary.set_steps.side_effect = lambda steps: setattr(rotary.get_current_position, "return_value",
                                                         rotary.get_current_position.return_value + steps)
    rotary.set_enabled.side_effect = lambda enabled: setattr(rotary.get_enabled, "return_value", enabled)
    args = SimpleNamespace(mode=mode, axis="both", host="localhost", port=4223)
    if broken:
        with pytest.raises(MotionHardwareError, match="OUT 0: no closure"):
            rig.run(args)
    else:
        rig.run(args)
    connection.disconnect.assert_called_once()
    if mode == "check":
        motor.set_enabled.assert_not_called()
        sender.connect.assert_not_called()
    else:
        sender.shutdown.assert_called_once()
        assert [call.args for call in sender.press_output.call_args_list] == [
            (ch, 80) for ch in range(1 if broken else 4)]
        assert not motor.get_enabled.return_value
        assert [call.args[0] for call in motor.set_steps.call_args_list] == ([] if broken else [-8, 8])
        assert [call.args[0] for call in rotary.set_steps.call_args_list] == ([] if broken else [-32, 32])
        assert not rotary.get_enabled.return_value
