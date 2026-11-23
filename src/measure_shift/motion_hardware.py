"""Optional Tinkerforge relay and digital input transport for the motion worker."""
from __future__ import annotations

import importlib
import math
import threading
from typing import TYPE_CHECKING, Any, Callable, Protocol

from .brick_usb import master_brick_access

if TYPE_CHECKING:
    from .bench_axis import SilentStepperBench


SILENT_STEPPER_V2_IDENTIFIER = 2166


class MotionHardwareError(RuntimeError):
    """An actionable motion transport or device error suitable for the UI."""


class MasterBrickAccessError(MotionHardwareError):
    """A local Master Brick is attached but inaccessible to this account."""


class MotionHardware(Protocol):
    def connect(self, host: str, port: int, digital_uid: str | None, output_uid: str) -> SilentStepperBench | None: ...
    def press_output(self, channel: int, duration_ms: int) -> None: ...
    def release_outputs(self) -> None: ...
    def read_inputs(self) -> tuple[bool, bool, bool, bool]: ...
    def shutdown(self) -> None: ...


class TinkerforgeMotionHardware:
    """Serialize input reads and finite relay pulses through Brick Daemon."""

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
        self._digital = None
        self._relay = None
        self._bench: SilentStepperBench | None = None
        self._industrial_relay = False
        self._ready = False
        self._digital_v2 = False
        self._connection_error: str | None = None
        self.digital_model = ""

    @property
    def connected(self) -> bool:
        return self._ready and self._connection_error is None

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
        if not self._ready:
            raise MotionHardwareError("Motion hardware is disconnected")

    def connect(self, host: str, port: int, digital_uid: str | None, output_uid: str) -> SilentStepperBench | None:
        if self._ipcon is not None:
            raise MotionHardwareError("Disconnect motion hardware before connecting again")
        uids = [output_uid] if digital_uid is None else [output_uid, digital_uid]
        if not host.strip() or any(not isinstance(uid, str) or not uid.strip() for uid in uids):
            raise MotionHardwareError("Host and Bricklet UIDs are required")
        if type(port) is not int or not 1 <= port <= 65535:
            raise MotionHardwareError("Brick Daemon port must be between 1 and 65535")
        if len(set(uids)) != len(uids):
            raise MotionHardwareError("Motion Bricklets must have different UIDs")
        try:
            ip_module = self._module_loader("tinkerforge.ip_connection")
            if digital_uid:
                digital_v1_class = self._module_loader("tinkerforge.bricklet_industrial_digital_in_4").BrickletIndustrialDigitalIn4
                digital_v2_class = self._module_loader("tinkerforge.bricklet_industrial_digital_in_4_v2").BrickletIndustrialDigitalIn4V2
            dual_relay = self._module_loader("tinkerforge.bricklet_dual_relay").BrickletDualRelay
            industrial_relay = self._module_loader("tinkerforge.bricklet_industrial_dual_relay").BrickletIndustrialDualRelay
        except (ImportError, OSError, AttributeError) as exc:
            raise MotionHardwareError(
                "Tinkerforge bindings are unavailable. Install measure-shift[motion] "
                f"in this environment. Details: {exc}"
            ) from exc
        ipcon = ip_module.IPConnection()
        self._ipcon = ipcon
        self._connection_error = None
        watched_uids = set(uids)
        discovery_lock = threading.Lock()
        discovering = True
        identities: dict[str, int] = {}
        discovery_failed = threading.Event()

        def enumerated(uid, connected_uid, position, hardware_version,
                       firmware_version, identifier, enumeration_type):
            with discovery_lock:
                if self._ipcon is not ipcon:
                    return
                if not discovering and uid not in watched_uids:
                    return
                if enumeration_type == ip_module.IPConnection.ENUMERATION_TYPE_DISCONNECTED:
                    if uid not in watched_uids:
                        identities.pop(uid, None)
                        return
                    self._connection_error = f"Motion Bricklet {uid} was disconnected; reconnect before moving"
                elif (not discovering
                      and (enumeration_type == ip_module.IPConnection.ENUMERATION_TYPE_CONNECTED
                           or identities.get(uid) != identifier)):
                    self._connection_error = f"Motion Bricklet {uid} was replaced or reconnected; reconnect before moving"
                else:
                    identities[uid] = identifier
                    return
                discovery_failed.set()

        def disconnected(reason):
            if self._ipcon is not ipcon:
                return
            self._connection_error = "Connection to Brick Daemon was lost; reconnect before moving"
            discovery_failed.set()

        try:
            ipcon.set_auto_reconnect(False)
            ipcon.set_timeout(self._request_timeout_s)
            ipcon.register_callback(ip_module.IPConnection.CALLBACK_ENUMERATE, enumerated)
            ipcon.register_callback(ip_module.IPConnection.CALLBACK_DISCONNECTED, disconnected)
            self._call(f"Cannot connect to Brick Daemon at {host}:{port}", ipcon.connect, host, port)
            self._call("Cannot discover motion Bricklets", ipcon.enumerate)
            # Enumeration has no completion callback. A stepper can be reported
            # after both the relay and digital input have already responded.
            discovery_failed.wait(self._enumeration_timeout_s)
            with discovery_lock:
                discovering = False
                steppers = [uid for uid, identifier in identities.items()
                            if identifier == SILENT_STEPPER_V2_IDENTIFIER]
                if len(steppers) > 1:
                    raise MotionHardwareError("Multiple Silent Steppers found. Connect only the intended axis and retry.")
                watched_uids.update(steppers)
                missing = watched_uids - identities.keys()
            if self._connection_error:
                raise MotionHardwareError(self._connection_error)
            if missing and False in master_brick_access(host):
                raise MasterBrickAccessError("A local Master Brick needs USB read/write permission.")
            if missing:
                raise MotionHardwareError("Motion Bricklet not found: " + ", ".join(sorted(missing)))
            relay_classes = {cls.DEVICE_IDENTIFIER: cls for cls in (dual_relay, industrial_relay)}
            relay_class = relay_classes.get(identities[output_uid])
            if relay_class is None:
                raise MotionHardwareError(f"UID {output_uid} is not a Dual Relay Bricklet")
            self._relay = relay_class(output_uid, ipcon)
            self._industrial_relay = relay_class is industrial_relay
            self._relay.set_response_expected_all(True)
            self.release_outputs()
            if digital_uid:
                digital_classes = {digital_v1_class.DEVICE_IDENTIFIER: digital_v1_class,
                                   digital_v2_class.DEVICE_IDENTIFIER: digital_v2_class}
                digital_class = digital_classes.get(identities[digital_uid])
                if digital_class is None:
                    raise MotionHardwareError(f"UID {digital_uid} is not an Industrial Digital In 4 Bricklet")
                self._digital = digital_class(digital_uid, ipcon)
                self._digital_v2 = digital_class is digital_v2_class
                self._digital.set_response_expected_all(True)
                self.digital_model = "Industrial Digital In 4 Bricklet" + (" 2.0" if self._digital_v2 else "")
            if self._digital is not None:
                self._check_digital_group()
            if steppers:
                from .bench_axis import SilentStepperBench
                stepper_class = self._module_loader("tinkerforge.bricklet_silent_stepper_v2").BrickletSilentStepperV2
                device = stepper_class(steppers[0], ipcon)
                device.set_response_expected_all(True)
                self._bench = SilentStepperBench(device)
                self._bench.stop_and_disable()
            self._ready = True
            self._require_connected()
            return self._bench
        except Exception as exc:
            try:
                self.shutdown()
            except Exception as cleanup:
                raise MotionHardwareError(f"Cannot initialize motion hardware: {exc}. Shutdown: {cleanup}") from exc
            if isinstance(exc, MotionHardwareError):
                raise
            raise MotionHardwareError(f"Cannot initialize motion hardware: {exc}") from exc

    def press_output(self, channel: int, duration_ms: int) -> None:
        self._require_connected()
        if (self._relay is None or type(channel) is not int or channel not in (0, 1)
                or type(duration_ms) is not int or not 1 <= duration_ms <= 5000):
            raise MotionHardwareError("Invalid joystick relay pulse")
        # The original Dual Relay numbers its coils 1 and 2. The Industrial
        # Dual Relay numbers them 0 and 1. Both release on the device timer.
        coil = channel if self._industrial_relay else channel + 1
        self._call("Cannot pulse joystick relay", self._relay.set_monoflop, coil, True, duration_ms)

    def release_outputs(self) -> None:
        if self._relay is None:
            return
        set_both = self._relay.set_value if self._industrial_relay else self._relay.set_state
        self._call("Cannot release joystick relays", set_both, False, False)

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
        if self._digital is None:
            return (False,) * 4
        self._check_digital_group()
        value = self._call("Cannot read joystick inputs", self._digital.get_value)
        if self._digital_v2:
            if len(value) != 4:
                raise MotionHardwareError("Digital input Bricklet returned an invalid channel count")
            return tuple(bool(channel) for channel in value)
        return tuple(bool(int(value) & (1 << channel)) for channel in range(4))

    def shutdown(self) -> None:
        ipcon = self._ipcon
        if ipcon is None:
            return
        errors = []
        try:
            self.release_outputs()
        except Exception as exc:
            errors.append(str(exc))
        try:
            if self._bench is not None:
                self._bench.stop_and_disable()
        except Exception as exc:
            errors.append(f"Cannot stop bench motor: {exc}")
        finally:
            try:
                ipcon.disconnect()
            except Exception as exc:
                errors.append(f"Cannot disconnect Brick Daemon: {exc}")
            self._ipcon = self._digital = self._relay = self._bench = None
            self._ready = False
            self._connection_error = None
        if errors:
            raise MotionHardwareError(". ".join(errors))
