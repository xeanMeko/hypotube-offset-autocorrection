"""Optional Tinkerforge transport, owned exclusively by the motion worker.

Importing this module does not import the SDK, open a connection, or power a
motor. Distances and velocities are in the selected driver microsteps. Electrical
input levels are returned unchanged; the controller handles mapping and polarity.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import threading
import time
from typing import Any, Callable, Protocol


class MotionHardwareError(RuntimeError):
    """An actionable motion transport or device error suitable for the UI."""


@dataclass(frozen=True)
class HardwareState:
    position: int = 0
    velocity: int = 0
    remaining: int = 0
    enabled: bool = False
    motor_current_ma: int = 0
    step_resolution: int = 8
    interpolation: bool = True
    fault: str | None = None
    input_voltage_mv: int | None = None


class MotionHardware(Protocol):
    def connect(self, host: str, port: int, stepper_uid: str, digital_uid: str) -> None: ...
    def read_inputs(self) -> tuple[bool, bool, bool, bool]: ...
    def read_state(self) -> HardwareState: ...
    def configure(self, motor_current_ma: int, step_resolution: int = 8, interpolation: bool = True) -> None: ...
    def enable(self) -> None: ...
    def disable(self) -> None: ...
    def move_steps(self, steps: int, velocity: int) -> None: ...
    def stop(self) -> None: ...
    def emergency_stop(self) -> None: ...
    def shutdown(self, timeout_s: float = 2.0) -> None: ...


class TinkerforgeMotionHardware:
    """Lazy adapter for Silent Stepper 2.0 and Industrial Digital In 4 v1/v2.

    The calling worker serializes all public operations. SDK callbacks only
    record device availability; they never issue movement commands. Finite
    relative moves avoid relying on a communication watchdog that these devices
    do not provide through their documented API.
    """

    def __init__(self, module_loader: Callable[[str], Any] = importlib.import_module,
                 *, enumeration_timeout_s: float = 2.0, request_timeout_s: float = .25):
        self._module_loader = module_loader
        self._enumeration_timeout_s = float(enumeration_timeout_s)
        self._request_timeout_s = float(request_timeout_s)
        if not math.isfinite(self._enumeration_timeout_s) or self._enumeration_timeout_s <= 0:
            raise ValueError("Enumeration timeout must be positive")
        if not math.isfinite(self._request_timeout_s) or self._request_timeout_s <= 0:
            raise ValueError("Request timeout must be positive")
        self._ipcon = None
        self._stepper = None
        self._digital = None
        self._digital_v2 = False
        self._connection_error: str | None = None
        self._watched_uids: set[str] = set()
        self._single_step_target: int | None = None
        self.stepper_model = ""
        self.digital_model = ""

    @property
    def connected(self) -> bool:
        return self._stepper is not None and self._digital is not None and self._connection_error is None

    def _call(self, description: str, function: Callable, *args):
        try:
            return function(*args)
        except MotionHardwareError:
            raise
        except Exception as exc:
            raise MotionHardwareError(f"{description}: {exc}") from exc

    def _require_connected(self) -> None:
        if self._connection_error:
            raise MotionHardwareError(self._connection_error)
        if self._stepper is None or self._digital is None:
            raise MotionHardwareError("Motion hardware is disconnected")

    def connect(self, host: str, port: int, stepper_uid: str, digital_uid: str) -> None:
        if self._ipcon is not None:
            raise MotionHardwareError("Disconnect motion hardware before connecting again")
        if not host.strip() or not stepper_uid.strip() or not digital_uid.strip():
            raise MotionHardwareError("Host and both Bricklet UIDs are required")
        if type(port) is not int or not 1 <= port <= 65535:
            raise MotionHardwareError("Brick Daemon port must be between 1 and 65535")
        if stepper_uid == digital_uid:
            raise MotionHardwareError("Stepper and digital input must have different UIDs")
        try:
            ip_module = self._module_loader("tinkerforge.ip_connection")
            stepper_class = self._module_loader("tinkerforge.bricklet_silent_stepper_v2").BrickletSilentStepperV2
            digital_v1_class = self._module_loader("tinkerforge.bricklet_industrial_digital_in_4").BrickletIndustrialDigitalIn4
            digital_v2_class = self._module_loader("tinkerforge.bricklet_industrial_digital_in_4_v2").BrickletIndustrialDigitalIn4V2
        except (ImportError, OSError, AttributeError) as exc:
            raise MotionHardwareError(
                "Tinkerforge bindings are unavailable. Install measure-shift[motion] "
                f"in this environment. Details: {exc}"
            ) from exc
        ipcon = ip_module.IPConnection()
        self._ipcon = ipcon
        self._connection_error = None
        self._watched_uids = {stepper_uid, digital_uid}
        identities: dict[str, int] = {}
        available = threading.Event()

        def enumerated(uid, connected_uid, position, hardware_version,
                       firmware_version, identifier, enumeration_type):
            if self._ipcon is not ipcon:
                return
            if uid not in self._watched_uids:
                return
            if enumeration_type == ip_module.IPConnection.ENUMERATION_TYPE_DISCONNECTED:
                self._connection_error = f"Motion Bricklet {uid} was disconnected; reconnect before moving"
                available.set()
                return
            if (self._stepper is not None
                    and (enumeration_type == ip_module.IPConnection.ENUMERATION_TYPE_CONNECTED
                         or identities.get(uid) != identifier)):
                self._connection_error = f"Motion Bricklet {uid} was replaced or reconnected; reconnect before moving"
                available.set()
                return
            identities[uid] = identifier
            if self._watched_uids.issubset(identities):
                available.set()

        def disconnected(reason):
            if self._ipcon is not ipcon:
                return
            self._connection_error = "Connection to Brick Daemon was lost; reconnect before moving"
            available.set()

        try:
            ipcon.set_auto_reconnect(False)
            ipcon.set_timeout(self._request_timeout_s)
            ipcon.register_callback(ip_module.IPConnection.CALLBACK_ENUMERATE, enumerated)
            ipcon.register_callback(ip_module.IPConnection.CALLBACK_DISCONNECTED, disconnected)
            self._call(f"Cannot connect to Brick Daemon at {host}:{port}", ipcon.connect, host, port)
            self._call("Cannot discover motion Bricklets", ipcon.enumerate)
            available.wait(self._enumeration_timeout_s)
            if self._connection_error:
                raise MotionHardwareError(self._connection_error)
            missing = self._watched_uids - identities.keys()
            if missing:
                raise MotionHardwareError("Motion Bricklet not found: " + ", ".join(sorted(missing)))
            if identities[stepper_uid] != stepper_class.DEVICE_IDENTIFIER:
                raise MotionHardwareError(f"UID {stepper_uid} is not a Silent Stepper Bricklet 2.0")
            digital_classes = {digital_v1_class.DEVICE_IDENTIFIER: digital_v1_class,
                               digital_v2_class.DEVICE_IDENTIFIER: digital_v2_class}
            digital_class = digital_classes.get(identities[digital_uid])
            if digital_class is None:
                raise MotionHardwareError(f"UID {digital_uid} is not an Industrial Digital In 4 Bricklet")
            self._stepper = stepper_class(stepper_uid, ipcon)
            self._digital = digital_class(digital_uid, ipcon)
            self._digital_v2 = digital_class is digital_v2_class
            self._stepper.set_response_expected_all(True)
            self._digital.set_response_expected_all(True)
            self.stepper_model = "Silent Stepper Bricklet 2.0"
            self.digital_model = "Industrial Digital In 4 Bricklet" + (" 2.0" if self._digital_v2 else "")
            initial = self.read_state()
            if initial.enabled or initial.velocity or initial.remaining:
                # Take ownership in a verified stopped, disabled state even if
                # another application previously powered or moved this axis.
                self._stop_and_disable(2.0, context="connection")
                ipcon.set_timeout(self._request_timeout_s)
            self._check_digital_group()
            # Never write current/resolution, enable power, or move on connect.
        except Exception as exc:
            self._stepper = self._digital = None
            self._ipcon = None
            try:
                ipcon.disconnect()
            except Exception:
                pass
            if isinstance(exc, MotionHardwareError):
                raise
            raise MotionHardwareError(f"Cannot initialize motion hardware: {exc}") from exc

    def _check_digital_group(self) -> None:
        if not self._digital_v2:
            group = self._call("Cannot read Digital In 4 grouping", self._digital.get_group)
            if tuple(group) != ("n", "n", "n", "n"):
                raise MotionHardwareError(
                    "Digital In 4 is grouped, so channels 0–3 may belong to another Bricklet. "
                    "Use Brick Viewer to restore its standalone group (n, n, n, n), then reconnect."
                )

    def read_inputs(self) -> tuple[bool, bool, bool, bool]:
        self._require_connected()
        # Verify on every read so another application's regrouping cannot
        # silently replace the physical contacts driving the linear axis.
        self._check_digital_group()
        value = self._call("Cannot read joystick inputs", self._digital.get_value)
        if self._digital_v2:
            if len(value) != 4:
                raise MotionHardwareError("Digital input Bricklet returned an invalid channel count")
            return tuple(bool(channel) for channel in value)
        return tuple(bool(int(value) & (1 << channel)) for channel in range(4))

    def read_state(self) -> HardwareState:
        self._require_connected()
        ss = self._stepper
        data = self._call("Cannot read stepper state", ss.get_all_data)
        if self._single_step_target is not None:
            # Firmware stepper_make_step_speedramp() performs a single step
            # synchronously but leaves STEPPER_STATE_STEPS and step_counter
            # unchanged. Thus remaining_steps can stay nonzero at rest.
            # Clear that state only after telemetry confirms the exact target
            # and zero velocity, then use fresh telemetry to verify the stop.
            if (data.current_position == self._single_step_target and data.current_velocity == 0
                    and data.remaining_steps != 0):
                self._call("Cannot finalize completed single step", ss.stop)
                data = self._call("Cannot verify completed single step", ss.get_all_data)
            if data.current_velocity == 0 and data.remaining_steps == 0:
                self._single_step_target = None
        resolution, interpolation = self._call("Cannot read step resolution", ss.get_step_configuration)
        if type(resolution) is not int or resolution not in range(9):
            raise MotionHardwareError("Stepper returned an unsupported step resolution")
        driver = self._call("Cannot read motor driver status", ss.get_driver_status)
        faults = []
        if driver.short_to_ground:
            faults.append("Motor driver short to ground")
        if driver.over_temperature == ss.OVER_TEMPERATURE_LIMIT:
            faults.append("Motor driver overtemperature limit")
        if driver.motor_stalled:
            faults.append("Motor stalled")
        if data.input_voltage < 8000:
            faults.append("Motor supply is below 8 V")
        return HardwareState(
            position=int(data.current_position), velocity=int(data.current_velocity),
            remaining=int(data.remaining_steps),
            enabled=bool(self._call("Cannot read motor power", ss.get_enabled)),
            motor_current_ma=int(self._call("Cannot read motor current", ss.get_motor_current)),
            step_resolution=1 << (8 - resolution), interpolation=bool(interpolation),
            fault="; ".join(faults) or None, input_voltage_mv=int(data.input_voltage),
        )

    def _require_idle(self, *, healthy: bool = False) -> HardwareState:
        state = self.read_state()
        if state.velocity or state.remaining:
            raise MotionHardwareError("Wait for the motor to stop before changing settings or starting another move")
        if healthy and state.fault:
            raise MotionHardwareError(state.fault)
        return state

    def configure(self, motor_current_ma: int, step_resolution: int = 8,
                  interpolation: bool = True) -> None:
        if type(motor_current_ma) is not int or not 360 <= motor_current_ma <= 1640:
            raise MotionHardwareError("Motor current must be an integer between 360 and 1640 mA within the motor rating")
        if type(step_resolution) is not int or step_resolution not in {1, 2, 4, 8, 16, 32, 64, 128, 256}:
            raise MotionHardwareError("Step resolution must be a power of two from 1 to 256")
        self._require_idle()
        ss = self._stepper
        self._call("Cannot set motor current", ss.set_motor_current, motor_current_ma)
        self._call("Cannot set step resolution", ss.set_step_configuration,
                   8 - (step_resolution.bit_length() - 1), bool(interpolation))
        self._call("Cannot set motion time base", ss.set_time_base, 1)
        self._call("Cannot set discrete motion", ss.set_speed_ramping, 0, 0)

    def enable(self) -> None:
        self._require_idle(healthy=True)
        self._call("Cannot enable motor power", self._stepper.set_enabled, True)

    def disable(self) -> None:
        self._require_idle()
        self._call("Cannot disable motor power", self._stepper.set_enabled, False)

    def move_steps(self, steps: int, velocity: int) -> None:
        if type(steps) is not int or not -(2 ** 31) <= steps < 2 ** 31 or steps == 0:
            raise MotionHardwareError("Move must contain a nonzero signed 32-bit number of steps")
        if type(velocity) is not int or not 1 <= velocity <= 65535:
            raise MotionHardwareError("Motor velocity must be an integer between 1 and 65535 steps/s")
        state = self._require_idle(healthy=True)
        if not state.enabled:
            raise MotionHardwareError("Enable motor power before moving")
        ss = self._stepper
        self._call("Cannot set motion time base", ss.set_time_base, 1)
        self._call("Cannot set discrete motion", ss.set_speed_ramping, 0, 0)
        self._call("Cannot set motor speed", ss.set_max_velocity, velocity)
        self._call("Cannot move motor", ss.set_steps, steps)
        target = (state.position + steps + 2 ** 31) % (2 ** 32) - 2 ** 31
        self._single_step_target = target if abs(steps) == 1 else None

    def stop(self) -> None:
        self._require_connected()
        self._request_stop("Cannot stop motor")

    def _request_stop(self, description: str) -> bool:
        """Request a stop unless the driver is already verified off and idle."""
        ss = self._stepper
        if self._call("Cannot read motor power before stopping", ss.get_enabled):
            self._call(description, ss.stop)
            return True
        # Silent Stepper V2 firmware rejects STOP in STEPPER_STATE_OFF with
        # INVALID_PARAMETER (-9). Use fresh telemetry instead of sending STOP
        # or enabling the driver just to make that command legal.
        data = self._call("Cannot verify disabled motor is stopped", ss.get_all_data)
        if data.current_velocity or data.remaining_steps:
            raise MotionHardwareError(f"{description}: motor power is disabled but motion is still reported")
        return False

    def emergency_stop(self) -> None:
        self._require_connected()
        self._call("Cannot brake motor", self._stepper.full_brake)

    def _stop_and_disable(self, timeout_s: float, *, context: str) -> None:
        """Verify standstill before removing power, within a request budget."""
        ipcon, ss = self._ipcon, self._stepper
        deadline = time.monotonic() + timeout_s
        # Try stopping even if a disconnection callback is pending, without
        # falsely declaring a disconnected motor to be stopped.
        ipcon.set_timeout(min(self._request_timeout_s, timeout_s / 3))
        if not self._request_stop(f"Cannot stop motor during {context}"):
            return
        while True:
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                raise MotionHardwareError(f"Motor did not confirm stopping before {context} timeout; power was not disabled")
            ipcon.set_timeout(min(self._request_timeout_s, remaining_time))
            velocity = self._call("Cannot verify motor stopped", ss.get_current_velocity)
            if velocity == 0:
                remaining_time = deadline - time.monotonic()
                if remaining_time <= 0:
                    raise MotionHardwareError(f"{context.capitalize()} timed out before disabling stopped motor power")
                ipcon.set_timeout(min(self._request_timeout_s, remaining_time))
                self._call("Cannot disable stopped motor power", ss.set_enabled, False)
                return
            time.sleep(min(.01, max(0, deadline - time.monotonic())))

    def shutdown(self, timeout_s: float = 2.0) -> None:
        """Stop, confirm zero velocity, then disable and always disconnect.

        If acknowledgement or stopping fails, do not disable a possibly rotating
        motor. Surface that failure instead of claiming power was removed.
        """
        if self._ipcon is None:
            return
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Shutdown timeout must be positive")
        ipcon, ss = self._ipcon, self._stepper
        error = None
        try:
            if ss is not None:
                self._stop_and_disable(timeout_s, context="shutdown")
        except Exception as exc:
            error = exc if isinstance(exc, MotionHardwareError) else MotionHardwareError(f"Cannot shut down motion hardware: {exc}")
        finally:
            try:
                ipcon.disconnect()
            except Exception as exc:
                error = error or MotionHardwareError(f"Cannot disconnect Brick Daemon: {exc}")
            self._ipcon = self._stepper = self._digital = None
            self._watched_uids = set()
            self._single_step_target = None
            self._connection_error = None
        if error:
            raise error
