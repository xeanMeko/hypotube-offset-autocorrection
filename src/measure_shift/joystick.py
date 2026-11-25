"""One logical left/right press is one finite machine nudge."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import time
from typing import TYPE_CHECKING, Callable, Protocol

from .motion_hardware import MotionHardware, MotionHardwareError

if TYPE_CHECKING:
    from .motion import MotionConfig


class Direction(IntEnum):
    LEFT = -1
    RIGHT = 1


@dataclass(frozen=True)
class JoystickState:
    # A contact-output adapter can use its own pulse/release and settling timer.
    busy: bool = False
    fault: str | None = None
    timeout_s: float = 5.0
    alignment_contact_confirmed: bool = False


class JoystickPort(Protocol):
    """Worker-owned output. Each pulse includes release and may finish asynchronously.

    Report busy until another press and a post-movement camera read are safe.
    read_inputs reports operator contacts separately from generated pulses.
    Stop, disable and shutdown must release any asserted outputs.
    """

    def connect(self) -> None: ...
    def read_inputs(self) -> tuple[bool, bool, bool, bool]: ...
    def read_state(self) -> JoystickState: ...
    def setup(self) -> None: ...
    def enable(self) -> None: ...
    def disable(self) -> None: ...
    def nudge(self, direction: Direction) -> None: ...
    def stop(self) -> None: ...
    def shutdown(self) -> None: ...


# Digital In 1 is wired to relay contact A0, Digital In 2 to A1.
RELAY_FOR_INPUT = {1: 0, 2: 1}
INPUT_DEBOUNCE_S = .03


@dataclass
class _AlignmentPulse:
    direction: Direction
    channel: int
    opposite_channel: int
    active_high: bool
    confirmation_deadline: float
    confirmed: bool = False
    dispatched: bool = False


class WiredJoystick:
    """Send bounded relay presses and mask them from operator input."""

    def __init__(self, hardware: MotionHardware, config: Callable[[], MotionConfig],
                 clock: Callable[[], float] = time.monotonic,
                 stop_requested: Callable[[], bool] = lambda: False):
        self._hardware = hardware
        self._config = config
        self._clock = clock
        self._stop_requested = stop_requested
        self._bench = None
        self._alignment: _AlignmentPulse | None = None
        self._alignment_fault: str | None = None
        self.raw_inputs = (False,) * 4
        self._echo_channel: int | None = None
        self._contact_held = False
        self._pulse_until = self._ready_at = 0.

    def connect(self) -> None:
        cfg = self._config()
        self._bench = self._hardware.connect(cfg.host, cfg.port, cfg.digital_uid or None, cfg.output_uid)

    def read_inputs(self) -> tuple[bool, bool, bool, bool]:
        cfg = self._config()
        levels = list(self._hardware.read_inputs())
        self.raw_inputs = tuple(levels)
        if not cfg.digital_uid:
            levels[cfg.left_channel] = levels[cfg.right_channel] = not cfg.active_high
        if self._alignment is not None:
            pulse = self._alignment
            self._observe_alignment(levels)
            levels[pulse.channel] = not pulse.active_high
            return tuple(levels)
        channel = self._echo_channel
        if channel is not None:
            # During a pulse the shared input can confirm a release, but it
            # cannot distinguish a new operator press from the relay closing.
            if self._clock() < self._pulse_until + INPUT_DEBOUNCE_S:
                self._contact_held = self._contact_held and levels[channel] == cfg.active_high
                if not self._contact_held:
                    levels[channel] = not cfg.active_high
            else:
                self._echo_channel = None
        return tuple(levels)

    def read_state(self) -> JoystickState:
        cfg = self._config()
        timeout = (cfg.pulse_ms + cfg.release_ms) / 1000 + 5.
        motor = self._bench.read_state() if self._bench is not None else JoystickState()
        pulse = self._alignment
        return JoystickState(
            busy=motor.busy or pulse is not None or self._clock() < self._ready_at,
            fault=motor.fault or self._alignment_fault, timeout_s=timeout,
            alignment_contact_confirmed=bool(pulse and pulse.confirmed and not pulse.dispatched),
        )

    def setup(self) -> None:
        self._release()
        if self._bench is not None:
            self._bench.setup()

    def enable(self) -> None:
        pass

    def disable(self) -> None:
        self.stop()

    def nudge(self, direction: Direction, *, alignment: bool = False) -> None:
        if not isinstance(direction, Direction):
            raise ValueError("A joystick pulse must be LEFT or RIGHT")
        cfg = self._config()
        if self._alignment is not None or self._clock() < self._ready_at:
            raise MotionHardwareError("Wait for the joystick contact to release")
        if {cfg.left_channel, cfg.right_channel} != RELAY_FOR_INPUT.keys():
            raise MotionHardwareError("The joystick output harness requires direction inputs 1 and 2")
        channel = cfg.right_channel if direction == Direction.RIGHT else cfg.left_channel
        verify_contact = alignment and self._bench is not None
        if verify_contact:
            if not cfg.digital_uid:
                self._alignment_fault = "Bench Align X requires a configured Digital In"
                return
            levels = self._hardware.read_inputs()
            for contact in (cfg.left_channel, cfg.right_channel):
                if levels[contact] == cfg.active_high:
                    self._alignment_fault = f"Digital In {contact} is already active before the alignment pulse"
                    return
        elif self._bench is not None:
            self._bench.prepare_nudge()
        if self._stop_requested():
            return
        pulse_started = self._clock()
        self._hardware.press_output(RELAY_FOR_INPUT[channel], cfg.pulse_ms)
        self._pulse_until = self._clock() + cfg.pulse_ms / 1000
        release_s = cfg.release_ms / 1000
        if cfg.digital_uid:
            release_s = max(release_s, INPUT_DEBOUNCE_S)
            self._echo_channel = channel
            self._contact_held = self.raw_inputs[channel] == cfg.active_high
        self._ready_at = self._pulse_until + release_s
        if verify_contact:
            opposite = cfg.left_channel if direction == Direction.RIGHT else cfg.right_channel
            self._alignment = _AlignmentPulse(direction, channel, opposite, cfg.active_high,
                                              pulse_started + cfg.pulse_ms / 1000)
            self._observe_alignment(self._hardware.read_inputs())
        elif self._bench is not None:
            if self._stop_requested():
                self.stop()
                return
            self._bench.nudge(direction)

    def _observe_alignment(self, levels) -> None:
        pulse = self._alignment
        if pulse is None or self._alignment_fault:
            return
        now = self._clock()
        active = levels[pulse.channel] == pulse.active_high
        if levels[pulse.opposite_channel] == pulse.active_high:
            self._alignment_fault = f"Unexpected closure on Digital In {pulse.opposite_channel} during Align X"
        elif not pulse.confirmed:
            if now > pulse.confirmation_deadline:
                self._alignment_fault = f"No relay closure detected on Digital In {pulse.channel} during Align X"
            elif active:
                pulse.confirmed = True
        elif now >= self._ready_at:
            if active:
                self._alignment_fault = f"Digital In {pulse.channel} did not release after the alignment pulse"
            elif pulse.dispatched:
                self._alignment = None

    def move_after_contact(self) -> None:
        pulse = self._alignment
        if pulse is None or not pulse.confirmed or pulse.dispatched or self._alignment_fault:
            return
        pulse.dispatched = True
        if self._stop_requested():
            return
        self._bench.prepare_nudge()
        if not self._stop_requested():
            self._bench.nudge(pulse.direction)

    pulse = nudge  # Compatibility with callers of the original relay port.

    def _release(self) -> None:
        self._alignment = None
        self._alignment_fault = None
        self._hardware.release_outputs()
        now = self._clock()
        if now < self._pulse_until:
            # An early Stop starts the same contact-settling interval as a
            # timed release. Keep filtering the relay's input during it.
            release_s = self._ready_at - self._pulse_until
            self._pulse_until = now
            self._ready_at = now + release_s

    def stop(self) -> None:
        try:
            self._release()
        finally:
            if self._bench is not None:
                self._bench.stop_and_disable()

    def shutdown(self) -> None:
        try:
            self._hardware.shutdown()
        finally:
            self._bench = None
            self._alignment = None
            self._alignment_fault = None
            self._echo_channel = None
            self._ready_at = self._pulse_until = 0.
