"""Two-Master bench test. Run directly with check, test, or listen."""
import argparse
from contextlib import ExitStack
import sys
import time

from measure_shift.bench_axis import SilentStepperBench
from measure_shift.joystick import Direction, QUAD_OUTPUT
from measure_shift.motion_hardware import MotionHardwareError, TinkerforgeMotionHardware


SENDER = "64tBhw"
RECEIVER = "6k2K8q"
RELAY = "2imB"
DIGITAL = "2gna"
MOTOR = "2eke"
DIRECTIONS = {channel: direction for direction, channel in QUAD_OUTPUT.items()}
PULSE_MS = 80
POLL_S = .005
RELEASE_S = .03
TIMEOUT_S = 2.0


def identify(uid, master, connection, *models):
    # get_identity is the same read-only request on both Bricklet generations.
    device = models[0](uid, connection)
    identity = device.get_identity()
    if identity.uid != uid or identity.connected_uid != master:
        raise MotionHardwareError(f"Expected {uid} on {master}, found {identity.uid} on {identity.connected_uid}")
    for model in models:
        if model.DEVICE_IDENTIFIER == identity.device_identifier:
            return device if model is models[0] else model(uid, connection)
    raise MotionHardwareError(f"Unexpected device type {identity.device_identifier} for {uid}")


def active_channel(digital):
    if hasattr(digital, "get_group") and tuple(digital.get_group()) != ("n",) * 4:
        raise MotionHardwareError(f"Digital In {DIGITAL} must use its standalone group")
    value = digital.get_value()
    if isinstance(value, int):
        value = [value & (1 << channel) for channel in range(4)]
    active = [channel for channel, level in enumerate(value) if level]
    if len(active) > 1:
        raise MotionHardwareError(f"Multiple simulator inputs active: {active}")
    return active[0] if active else None


def await_idle(read, timeout=TIMEOUT_S):
    deadline = time.monotonic() + timeout
    idle_since = None
    while time.monotonic() < deadline:
        if read() is not None:
            idle_since = None
        elif idle_since is None:
            idle_since = time.monotonic()
        elif time.monotonic() - idle_since >= RELEASE_S:
            return
        time.sleep(POLL_S)
    raise MotionHardwareError("Timed out waiting for all four contacts to release")


def receive_press(read, bench, stepper, axis, *, expected=None):
    """Consume one contact closure, its release, and the corresponding finite move."""
    deadline = time.monotonic() + TIMEOUT_S
    while (channel := read()) is None:
        if expected is not None and time.monotonic() >= deadline:
            raise MotionHardwareError(f"OUT {expected}: no closure detected on IN {expected}")
        time.sleep(POLL_S)
    if expected is not None and channel != expected:
        raise MotionHardwareError(f"OUT {expected}: expected IN {expected}, observed IN {channel}")

    direction = DIRECTIONS[channel]
    motor_directions = ({Direction.RIGHT: Direction.RIGHT, Direction.LEFT: Direction.LEFT}
                        if axis == "x" else {Direction.UP: Direction.RIGHT, Direction.DOWN: Direction.LEFT})
    motor_direction = motor_directions.get(direction)
    before = stepper.get_current_position()
    delta = 0
    if motor_direction is not None:
        bench.prepare_nudge()
        bench.nudge(motor_direction)
        delta = -int(motor_direction) * bench.STEPS_PER_NUDGE

    def read_release():
        active = read()
        if active is not None and active != channel:
            raise MotionHardwareError(f"Unexpected IN {active} before IN {channel} finished releasing")
        return active

    await_idle(read_release)
    deadline = time.monotonic() + TIMEOUT_S
    while True:
        state = bench.read_state()
        if state.fault:
            raise MotionHardwareError(state.fault)
        if not state.busy:
            break
        if time.monotonic() >= deadline:
            raise MotionHardwareError(f"Stepper remained busy after IN {channel}")
        if read() is not None:
            raise MotionHardwareError("New contact while previous nudge was still in progress")
        time.sleep(POLL_S)
    # The driver's signed position counter wraps at 32 bits.
    observed = (stepper.get_current_position() - before + 2**31) % 2**32 - 2**31
    if observed != delta:
        raise MotionHardwareError(f"IN {channel}: expected driver delta {delta}, observed {observed}")
    print(f"IN {channel} {direction.name}: released, {axis.upper()} driver delta {observed:+d}", flush=True)


def run(args):
    from tinkerforge.bricklet_industrial_digital_in_4 import BrickletIndustrialDigitalIn4
    from tinkerforge.bricklet_industrial_digital_in_4_v2 import BrickletIndustrialDigitalIn4V2
    from tinkerforge.bricklet_industrial_quad_relay import BrickletIndustrialQuadRelay
    from tinkerforge.bricklet_industrial_quad_relay_v2 import BrickletIndustrialQuadRelayV2
    from tinkerforge.bricklet_silent_stepper_v2 import BrickletSilentStepperV2
    from tinkerforge.ip_connection import IPConnection

    with ExitStack() as cleanup:
        connection = IPConnection()
        connection.set_auto_reconnect(False)
        connection.set_timeout(.5)
        connection.connect(args.host, args.port)
        cleanup.callback(connection.disconnect)
        identify(RELAY, SENDER, connection, BrickletIndustrialQuadRelayV2, BrickletIndustrialQuadRelay)
        digital = identify(DIGITAL, RECEIVER, connection, BrickletIndustrialDigitalIn4V2, BrickletIndustrialDigitalIn4)
        stepper = identify(MOTOR, RECEIVER, connection, BrickletSilentStepperV2)
        bench = SilentStepperBench(stepper)

        def read():
            return active_channel(digital)

        print(f"Sender {SENDER}: relay {RELAY}. Receiver {RECEIVER}: inputs {DIGITAL}, stepper {MOTOR}.")
        channel = read()
        state = bench.read_state()
        print(f"Active input: {channel}. Stepper busy: {state.busy}. Fault: {state.fault}.", flush=True)
        if args.mode == "check":
            return
        if state.fault or state.busy:
            raise MotionHardwareError(state.fault or "Stepper is already moving")
        await_idle(read)
        stepper.set_response_expected_all(True)
        cleanup.callback(bench.stop_and_disable)
        bench.stop_and_disable()
        # Use the existing bench speed and step resolution, retaining this driver's current limit.
        current = stepper.get_motor_current()
        stepper.set_step_configuration(stepper.STEP_RESOLUTION_32, True)
        stepper.set_time_base(1)
        stepper.set_speed_ramping(0, 0)
        stepper.set_max_velocity(200)
        print(f"Motor axis {args.axis.upper()}, {bench.STEPS_PER_NUDGE} microsteps/press, {current} mA.", flush=True)
        if args.mode == "listen":
            print("Listening for relay contacts. Press Ctrl-C to stop.", flush=True)
            while True:
                receive_press(read, bench, stepper, args.axis)

        sender = TinkerforgeMotionHardware()
        sender.connect(args.host, args.port, None, RELAY, quad_relay=True)
        cleanup.callback(sender.shutdown)
        for direction, output in QUAD_OUTPUT.items():
            await_idle(read)
            print(f"Pulsing OUT {output} {direction.name}", flush=True)
            sender.press_output(output, PULSE_MS)
            receive_press(read, bench, stepper, args.axis, expected=output)
    print("PASS: all four contacts and driver deltas matched.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "test", "listen"),
                        help="read identities and status, run four timed pulses, or receive pulses from the app")
    parser.add_argument("--axis", choices=("x", "y"), default="x", help="axis represented by the one motor")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=4223)
    try:
        run(parser.parse_args())
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Bench test failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
