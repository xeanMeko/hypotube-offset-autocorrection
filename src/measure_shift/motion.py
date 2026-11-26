"""Finite joystick motion and camera-feedback alignment, independent of Qt.

The controller is deterministic; only MotionWorker talks to a hardware adapter.
Pixels remain camera pixels throughout. Neither jogging nor alignment modifies
an image or the reference used by the measurement engine.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, fields, replace
import math
import threading
import time
from typing import Callable

from .direct_alignment import AxisSearch
from .joystick import INPUT_DEBOUNCE_S, Direction, JoystickPort, JoystickState, WiredJoystick
from .motion_hardware import MasterBrickAccessError, TinkerforgeMotionHardware
from .workers import LatestValue, MeasurementWorker


REFERENCE_CAPTURE_INTERVAL_S = 1.0
KEY_DIRECTIONS = (Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN)


@dataclass(frozen=True)
class MotionConfig:
    host: str = "localhost"
    port: int = 4223
    digital_uid: str = "27s4"
    left_channel: int = 2
    right_channel: int = 1
    reference_channel: int | None = 0
    align_channel: int | None = 3
    active_high: bool = True
    repeat_hz: float = 2.0
    calibration_presses: int = 1
    calibration_source_id: str = ""
    calibration_shape: tuple[int, int] | None = None

    output_uid: str = "2j7T"
    pulse_ms: int = 80
    release_ms: int = 100
    pixels_per_right_press: float | None = None

    # Profiles written before relay selection used the dual-relay harness.
    quad_relay: bool = False

    @classmethod
    def quad_defaults(cls) -> MotionConfig:
        return cls(quad_relay=True, output_uid="2imB")

    def validated(self) -> MotionConfig:
        if type(self.quad_relay) is not bool:
            raise ValueError("Invalid relay type")
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("Brick Daemon host is required")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("Brick Daemon port must be between 1 and 65535")
        if not isinstance(self.digital_uid, str):
            raise ValueError("Digital In UID must be text")
        uids = [self.output_uid, self.digital_uid] if self.digital_uid else [self.output_uid]
        if not all(isinstance(v, str) and v.strip() for v in uids):
            raise ValueError("Enter a UID for each required motion Bricklet")
        if len(set(uids)) != len(uids):
            raise ValueError("Motion Bricklets must have different UIDs")
        if type(self.pulse_ms) is not int or not 1 <= self.pulse_ms <= 5000:
            raise ValueError("Joystick press duration must be between 1 and 5000 ms")
        if type(self.release_ms) is not int or not 1 <= self.release_ms <= 10000:
            raise ValueError("Joystick release interval must be between 1 and 10000 ms")
        if any(type(v) is not int or v not in range(4) for v in (self.left_channel, self.right_channel)):
            raise ValueError("Joystick channels must be between 0 and 3")
        if self.left_channel == self.right_channel:
            raise ValueError("Left and right must use different digital inputs")
        if self.reference_channel is not None and (type(self.reference_channel) is not int or self.reference_channel not in range(4)):
            raise ValueError("Reference capture channel must be disabled or between 0 and 3")
        if not self.quad_relay and self.reference_channel in (self.left_channel, self.right_channel):
            raise ValueError("Reference capture must use a different digital input from left and right")
        if self.align_channel is not None:
            if type(self.align_channel) is not int or self.align_channel not in range(4):
                raise ValueError("Alignment channel must be disabled or between 0 and 3")
            if self.quad_relay and self.align_channel == self.reference_channel:
                raise ValueError("Align XY and reference capture must use different digital inputs")
            if not self.quad_relay and self.align_channel in (self.left_channel, self.right_channel, self.reference_channel):
                raise ValueError("Align X must use a different digital input from left, right and reference capture")
        if type(self.active_high) is not bool:
            raise ValueError("Input polarity must be a boolean")
        if (isinstance(self.repeat_hz, bool) or not isinstance(self.repeat_hz, (int, float)) or
                not math.isfinite(self.repeat_hz)):
            raise ValueError("Hold repeat frequency must be finite")
        if not .1 <= self.repeat_hz <= 10:
            raise ValueError("Hold repeat frequency must be between 0.1 and 10 Hz")
        if type(self.calibration_presses) is not int or not 1 <= abs(self.calibration_presses) <= 1_000_000:
            raise ValueError("Calibration needs a nonzero integer within ±1000000 joystick presses")
        scale = self.pixels_per_right_press
        if scale is not None and (isinstance(scale, bool) or not isinstance(scale, (int, float))
                                  or not math.isfinite(scale) or scale == 0):
            raise ValueError("Pixel calibration scale must be finite and nonzero")
        if not isinstance(self.calibration_source_id, str):
            raise ValueError("Calibration source identifier must be text")
        if self.calibration_shape is not None and (len(self.calibration_shape) != 2 or any(type(v) is not int or v <= 0 for v in self.calibration_shape)):
            raise ValueError("Calibration image geometry must contain two positive integers")
        return self

    def without_press_scale(self) -> MotionConfig:
        return replace(self, pixels_per_right_press=None, calibration_source_id="", calibration_shape=None)

    def with_press_scale(self, pixels: float, context: MotionContext) -> MotionConfig:
        return replace(self, pixels_per_right_press=pixels, calibration_source_id=context.source_id,
                       calibration_shape=context.shape).validated()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> MotionConfig:
        if not isinstance(value, dict):
            raise ValueError("Motion settings must be an object")
        names = {f.name for f in fields(cls)}
        kwargs = {key: item for key, item in value.items() if key in names}
        if "pixels_per_right_press" not in kwargs and value.get("bench_axis") is False:
            kwargs["pixels_per_right_press"] = value.get("factory_pixels_per_press")
        if kwargs.get("quad_relay"):
            kwargs.setdefault("output_uid", cls.quad_defaults().output_uid)
        elif kwargs.get("output_uid") == "23GY":
            kwargs["output_uid"] = cls.output_uid
            kwargs["pixels_per_right_press"] = None
        if "align_channel" not in kwargs and cls.align_channel in (
                kwargs.get("left_channel", cls.left_channel),
                kwargs.get("right_channel", cls.right_channel),
                kwargs.get("reference_channel", cls.reference_channel)):
            # Preserve existing wiring when an older profile already uses
            # channel 3 for another action.
            kwargs["align_channel"] = None
        if "reference_channel" not in kwargs and cls.reference_channel in (
                kwargs.get("left_channel", cls.left_channel),
                kwargs.get("right_channel", cls.right_channel),
                kwargs.get("align_channel", cls.align_channel)):
            # Preserve older wiring that already assigns the new default input.
            kwargs["reference_channel"] = None
        if not kwargs.get("quad_relay") and type(kwargs.get("reference_channel")) is int and kwargs["reference_channel"] in (
                kwargs.get("left_channel", cls.left_channel), kwargs.get("right_channel", cls.right_channel)):
            # Older profiles allowed capture to share a joystick contact.
            # Preserve the joystick wiring and disable the shared capture trigger.
            kwargs["reference_channel"] = None
        if kwargs.get("pixels_per_right_press") is None:
            kwargs["calibration_source_id"] = ""
            kwargs["calibration_shape"] = None
        elif kwargs.get("calibration_shape") is not None:
            kwargs["calibration_shape"] = tuple(kwargs["calibration_shape"])
        if "calibration_presses" not in kwargs and "calibration_steps" in value:
            steps = value["calibration_steps"]
            if type(steps) is not int or not 1 <= abs(steps) <= 1_000_000:
                raise ValueError("Invalid legacy calibration move")
            # The old driver-step distance cannot determine relay travel.
            # Preserve the test direction with a single relay press.
            kwargs["calibration_presses"] = -1 if (steps < 0) != bool(value.get("invert_direction", True)) else 1
        return cls(**kwargs).validated()


@dataclass(frozen=True)
class MotionContext:
    source_id: str = ""
    source_kind: str = ""
    shape: tuple[int, int] | None = None
    reference_id: str | None = None
    config_id: str | None = None
    connected: bool = False
    paused: bool = False
    tolerance_px: float = 1.0
    max_feedback_age_s: float = 1.0
    settling_s: float = .2

    @property
    def key(self):
        return (self.source_id, self.source_kind, self.shape, self.reference_id,
                self.config_id, self.connected, self.paused, self.tolerance_px)

    @property
    def live_camera_ready(self) -> bool:
        return (self.connected and not self.paused and self.source_kind.lower() in {"ids", "basler"}
                and self.shape is not None and self.config_id is not None)

    @property
    def camera_ready(self) -> bool:
        return self.live_camera_ready and self.reference_id is not None


@dataclass(frozen=True)
class MotionFeedback:
    frame_id: str
    reference_id: str | None
    config_id: str
    dx: float
    usable: bool
    received_at: float
    read_started_at: float | None = None
    freshness_generation: int = 0
    dy: float = 0.0
    validity: str = ""

    @classmethod
    def from_pair(cls, pair) -> MotionFeedback:
        return cls(pair.frame.frame_id, pair.reference.reference_id, pair.config.config_id,
                   pair.result.dx, pair.result.usable, pair.frame.received_at,
                   getattr(pair.frame, "read_started_at", None), getattr(pair.frame, "freshness_generation", 0),
                   dy=pair.result.dy, validity=pair.result.validity)


@dataclass(frozen=True)
class MotionAction:
    kind: str
    direction: Direction | None = None
    after: float = 0.0
    alignment: bool = False


@dataclass(frozen=True)
class ReferenceCaptureRequest:
    channel: int
    triggered_at: float
    config_revision: int


@dataclass(frozen=True)
class MotionStatus:
    connected: bool = False
    setup_applied: bool = False
    armed: bool = False
    phase: str = "disconnected"
    message: str = "Connect motion hardware to begin"
    inputs4: tuple[bool, bool, bool, bool] = (False, False, False, False)
    corrections: int = 0
    dx: float | None = None
    config: MotionConfig = MotionConfig()
    config_revision: int = 0
    reference_capture: ReferenceCaptureRequest | None = None
    connection_revision: int = 0
    busy: bool = False
    usb_access_needed: bool = False


class MotionController:
    """A tick-driven state machine producing finite hardware actions."""

    SETTLE_S = .20
    MAX_OPERATION_S = 60.0
    MAX_CORRECTIONS = 100
    FEEDBACK_RETRY_S = 3.0
    UNCERTAIN_ALIGNMENT_MIN_OFFSET_PX = 10.0

    def __init__(self, config: MotionConfig | None = None):
        self.config = (config or MotionConfig()).validated()
        self.config_revision = 0
        self.connection_revision = 0
        self.usb_access_needed = False
        self.context = MotionContext()
        self.connected = False
        self.setup_applied = False
        self.armed = False
        self.phase = "disconnected"
        self.message = "Connect motion hardware to begin"
        self.state = JoystickState()
        self.inputs4 = (False, False, False, False)
        self.reference_capture: ReferenceCaptureRequest | None = None
        self._reference_level: bool | None = None
        self._last_reference_capture_at: float | None = None
        self._align_level: bool | None = None
        self.keyboard = (False,) * 4
        self._saw_conflict = False
        self._manual_activity = False
        self._debounced = self.inputs4
        self._candidate_inputs = self.inputs4
        self._input_since = [0.0] * 4
        self._neutral_required = True
        self._direction = 0
        self._next_nudge_at = 0.0
        self._actions: list[MotionAction] = []
        self._move = None
        self._operation: str | None = None
        self._stage = ""
        self._pending_operation: tuple[str, int] | None = None
        self._op_started = 0.0
        self._fresh_cutoff = 0.0
        self._required_generation: int | None = None
        self._last_consumed_frame: str | None = None
        self._feedback: MotionFeedback | None = None
        self._feedback_retry_started_at: float | None = None
        self._calibration_start_dx = 0.0
        self._calibration_presses = 0
        self._presses_sent = 0
        self._corrections = 0
        self._confirmations = 0
        self._alignment_search: AxisSearch | None = None
        self._stopping = False

    def configure(self, config: MotionConfig, revision: int = 0) -> None:
        config = config.validated()
        old = self.config
        self.usb_access_needed = False
        if self._operation or self._pending_operation:
            self.cancel("Motion settings changed", disarm=False)
        press_changed = any(getattr(old, name) != getattr(config, name) for name in
                            ("host", "port", "quad_relay", "output_uid", "left_channel", "right_channel", "pulse_ms"))
        if press_changed and config.pixels_per_right_press == old.pixels_per_right_press:
            config = config.without_press_scale()
        if config.pixels_per_right_press is not None and not config.calibration_source_id and self.context.shape is not None:
            config = replace(config, calibration_source_id=self.context.source_id, calibration_shape=self.context.shape)
        controls_changed = press_changed or any(getattr(old, name) != getattr(config, name) for name in
                                                ("active_high", "release_ms"))
        if config.reference_channel != old.reference_channel:
            self._reference_level = None
            self.reference_capture = None
        if config.align_channel != old.align_channel:
            self._align_level = None
        self.config = config
        self.config_revision = revision
        if controls_changed and self.connected:
            self.cancel("Joystick settings changed. Release both directions", disarm=False)
        self._invalidate_camera_scale()

    def _invalidate_camera_scale(self):
        if self.config.pixels_per_right_press is None or not self.context.source_id or self.context.shape is None:
            return
        if (self.config.calibration_source_id != self.context.source_id or
                self.config.calibration_shape != self.context.shape):
            self.config = self.config.without_press_scale()
            self.message = "Camera or image geometry changed; calibrate image pixels again"

    def set_context(self, context: MotionContext) -> None:
        if not math.isfinite(context.tolerance_px) or context.tolerance_px < 0:
            raise ValueError("Horizontal tolerance must be finite and nonnegative")
        if not math.isfinite(context.max_feedback_age_s) or context.max_feedback_age_s <= 0:
            raise ValueError("Measurement freshness allowance must be positive")
        if not math.isfinite(context.settling_s) or context.settling_s < 0:
            raise ValueError("Camera settling time must be finite and nonnegative")
        if context.key != self.context.key and (self._operation or self._pending_operation):
            self.cancel("Camera, reference or matching context changed", disarm=False)
        self.context = context
        self._invalidate_camera_scale()

    def set_keyboard(self, left: bool, right: bool, up=False, down=False, *, conflict=False, activity=False) -> None:
        self.keyboard = bool(left), bool(right), bool(up), bool(down)
        after = self._logical_inputs(raw=True)
        if conflict or sum(after) > 1:
            self._saw_conflict = True
        if (activity or any(after)) and self._operation:
            self._manual_activity = True

    def submit_feedback(self, feedback: MotionFeedback | None) -> None:
        self._feedback = feedback

    def on_connected(self, state: JoystickState) -> None:
        self.connected = True
        self.usb_access_needed = False
        self._reference_level = None
        self.reference_capture = None
        self._align_level = None
        self.state = state
        self.setup_applied = False
        self.armed = False
        self._neutral_required = True
        self._direction = 0
        self._move = None
        self._operation = None
        self._pending_operation = None
        self._stopping = False
        self.phase = "connected"
        self.message = "Connected. Enable movement with the controls released"

    def on_disconnected(self, message="Motion hardware disconnected", *, usb_access_needed=False) -> None:
        self.usb_access_needed = usb_access_needed
        self.connected = self.setup_applied = self.armed = False
        self.state = JoystickState()
        self._reference_level = None
        self.reference_capture = None
        self._align_level = None
        self._operation = self._pending_operation = self._move = None
        self._actions.clear()
        self._stopping = False
        self._neutral_required = True
        self.phase, self.message = "disconnected", message

    def on_setup_applied(self):
        self.setup_applied = True
        self.armed = False
        self._operation = self._pending_operation = self._move = None
        self._actions.clear()
        self._stopping = False
        self._neutral_required = True
        self.phase, self.message = "connected", "Setup applied. Enable movement with joystick and arrow keys released"

    def arm(self, *, wait_for_neutral: bool = False):
        if not self.connected or not self.setup_applied:
            self.message = "Connect hardware and apply setup before enabling movement"
            return
        if self.state.fault:
            self.message = self.state.fault
            return
        if any(self._logical_inputs(raw=True)):
            held = []
            for name, channel in (("Left", self.config.left_channel), ("Right", self.config.right_channel)):
                if not self.config.quad_relay and self.inputs4[channel] == self.config.active_high:
                    level = "HIGH" if self.inputs4[channel] else "LOW"
                    held.append(f"{name} joystick input (channel {channel}: {level})")
            held.extend(f"{name} arrow key" for name, pressed in zip(("Left", "Right", "Up", "Down"), self.keyboard) if pressed)
            self.message = "Release controls before moving: " + "; ".join(held)
            if not self.config.quad_relay and not self.config.active_high and any(
                    not self.inputs4[channel] for channel in (self.config.left_channel, self.config.right_channel)):
                self.message += ". If joystick inputs are LOW when released, check Active high"
            if not wait_for_neutral:
                return
        if self._move or self._stopping or self.state.busy:
            self.message = "Wait for the nudge to finish, then enable movement"
            return
        self.armed = True
        self._neutral_required = wait_for_neutral
        self._direction = 0
        if wait_for_neutral:
            self.phase = "waiting"
            if not any(self._logical_inputs(raw=True)):
                self.message = "Connected · waiting for released controls"
        else:
            self.phase, self.message = "ready", "Ready · hold to nudge one move at a time"
        self._actions.append(MotionAction("enable"))

    def request_align(self):
        self._pending_operation = ("align_direct", 0)

    request_align_direct = request_align

    def request_calibration(self, presses: int):
        if type(presses) is not int or not 1 <= abs(presses) <= 1_000_000:
            raise ValueError("Calibration needs a nonzero signed number of joystick presses (±1000000)")
        self._pending_operation = ("calibration", presses)

    def cancel(self, message="Motion stopped; release both directions before the next press", *, disarm=False):
        # Recoverable operation failures stop motion without requiring another
        # Enable click. Hardware faults and setup changes explicitly disarm.
        self._operation = self._pending_operation = None
        self._alignment_search = None
        self._feedback_retry_started_at = None
        self._confirmations = 0
        self._neutral_required = True
        self._direction = 0
        if disarm:
            self.armed = False
        self.message = message
        if self.connected:
            self._actions = [MotionAction("stop")]
            self._stopping = True
            self.phase = "stopping"
        else:
            self.phase = "disconnected"

    def acknowledge_freshness(self, generation: int):
        if type(generation) is not int or generation <= 0:
            self.cancel("Camera freshness barrier is unavailable")
            return
        self._required_generation = generation

    def _logical_inputs(self, *, raw=False):
        if self.config.quad_relay:
            return self.keyboard
        inputs = self.inputs4 if raw else self._debounced
        left = inputs[self.config.left_channel] == self.config.active_high
        right = inputs[self.config.right_channel] == self.config.active_high
        return left or self.keyboard[0], right or self.keyboard[1]

    def _freshness(self, now):
        self._fresh_cutoff = now + max(self.SETTLE_S, self.context.settling_s)
        if self._stage == "postmove" and (
                self._operation != "calibration" or self._presses_sent < abs(self._calibration_presses)):
            self._fresh_cutoff = max(self._fresh_cutoff, self._next_nudge_at)
        self._required_generation = None
        self._last_consumed_frame = None
        self._actions.append(MotionAction("freshness", after=self._fresh_cutoff))

    def _press(self, direction: Direction, now: float, purpose: str):
        self._move = (purpose, now, self.state.timeout_s)
        self._actions.append(MotionAction("pulse", direction, alignment=purpose == "align"))
        self._next_nudge_at = now + 1 / self.config.repeat_hz
        self.phase = {"jog": "jogging", "align": "aligning", "calibration": "calibrating"}[purpose]

    def _start_operation(self, kind, presses, now):
        if self._operation:
            return
        if not self.armed or not self.setup_applied:
            self.message = "Enable movement before using camera feedback"
            return
        if self._move or self.state.busy or self._stopping:
            self.message = "Wait for the current nudge to finish"
            return
        if any(self._logical_inputs(raw=True)):
            self.message = "Release joystick and arrow keys before using camera feedback"
            return
        if not self.context.live_camera_ready:
            self.message = "A running, unpaused IDS or Basler camera is required"
            return
        if kind == "align_direct" and not self.context.camera_ready:
            self.message = "A running, unpaused IDS or Basler camera and a reference are required"
            return
        self._operation, self._stage = kind, "baseline"
        self._op_started = now
        self._feedback_retry_started_at = None
        self._calibration_presses = presses
        self._presses_sent = 0
        self._corrections = self._confirmations = 0
        self._alignment_search = None
        self._feedback = None
        self._neutral_required = True
        self.phase = "calibrating" if kind == "calibration" else "aligning"
        self.message = "Waiting for a fresh, settled camera measurement"
        self._freshness(now)

    def _feedback_usable(self, feedback):
        if not all(math.isfinite(value) for value in (feedback.dx, feedback.dy)):
            return False
        if feedback.usable:
            return True
        # Permit uncertain estimates for coarse alignment only. Calibration
        # still needs valid measurements to establish the displacement per press.
        threshold = max(self.UNCERTAIN_ALIGNMENT_MIN_OFFSET_PX, self.context.tolerance_px)
        return (self._operation == "align_direct" and feedback.validity == "uncertain"
                and max(abs(feedback.dx), abs(feedback.dy)) > threshold)

    def _retry_operation_feedback(self, now, reason):
        # Count elapsed time, not controller ticks or repeated reads of one
        # frame. New bad frames and post-move freshness barriers must not
        # extend this recovery window.
        if self._feedback_retry_started_at is None:
            self._feedback_retry_started_at = now
        self._confirmations = 0
        if now - self._feedback_retry_started_at >= self.FEEDBACK_RETRY_S:
            self.cancel(f"{reason}; no usable camera feedback within {self.FEEDBACK_RETRY_S:g} seconds")
        else:
            self.message = f"{reason}; retrying camera feedback"

    def _consume_operation_feedback(self, now):
        feedback = self._feedback
        if (feedback is None or self._required_generation is None or
                feedback.freshness_generation < self._required_generation or
                feedback.read_started_at is None or feedback.read_started_at < self._fresh_cutoff or
                feedback.received_at < self._fresh_cutoff or
                feedback.reference_id != self.context.reference_id or feedback.config_id != self.context.config_id):
            if self._feedback_retry_started_at is not None:
                self._retry_operation_feedback(now, "Fresh camera measurement is unavailable")
            elif now > self._fresh_cutoff + max(3.0, self.context.max_feedback_age_s):
                self.cancel("Fresh camera measurement timed out")
            return None
        if not self._feedback_usable(feedback):
            if (self._operation == "align_direct" and feedback.validity == "uncertain"
                    and math.isfinite(feedback.dx) and math.isfinite(feedback.dy)):
                self._retry_operation_feedback(now, "Camera match is uncertain at low offset; a valid match is required")
            else:
                self._retry_operation_feedback(now, "Camera match is invalid or uncertain")
            return None
        if now - feedback.received_at > self.context.max_feedback_age_s:
            self._retry_operation_feedback(now, "Camera measurement is stale")
            return None
        if feedback.frame_id == self._last_consumed_frame:
            return None
        self._feedback_retry_started_at = None
        self._last_consumed_frame = feedback.frame_id
        return feedback

    def _process_operation(self, now):
        if not self._operation:
            return
        if now - self._op_started > self.MAX_OPERATION_S:
            self.cancel("Camera-feedback motion reached its 60 second limit")
            return
        if self._move:
            feedback = self._feedback
            if (feedback is None or feedback.received_at < self._op_started or
                    feedback.reference_id != self.context.reference_id or feedback.config_id != self.context.config_id or
                    not self._feedback_usable(feedback) or now - feedback.received_at > self.context.max_feedback_age_s):
                self._retry_operation_feedback(now, "Camera feedback became invalid, uncertain at low offset, or stale during movement")
            else:
                if self._feedback_retry_started_at is not None:
                    self.message = "Camera feedback recovered; continuing current move"
                self._feedback_retry_started_at = None
            return
        if self.state.busy:
            self.cancel("Joystick output became busy while waiting for camera feedback")
            return
        feedback = self._consume_operation_feedback(now)
        if feedback is None or not self._operation:
            return
        if self._operation == "calibration":
            if self._stage == "baseline":
                self._calibration_start_dx = feedback.dx
            if self._presses_sent < abs(self._calibration_presses):
                direction = Direction.RIGHT if self._calibration_presses > 0 else Direction.LEFT
                self._presses_sent += 1
                self._stage = "moving"
                self.message = f"Calibration press {self._presses_sent}/{abs(self._calibration_presses)}"
                self._press(direction, now, "calibration")
            else:
                delta_pixels = feedback.dx - self._calibration_start_dx
                if not math.isfinite(delta_pixels) or abs(delta_pixels) < 2:
                    self.cancel("Calibration needs at least 2 pixels of observed X movement")
                    return
                scale = delta_pixels / self._calibration_presses
                self.config = self.config.with_press_scale(scale, self.context)
                self._operation = None
                self.phase = "ready"
                self.message = f"Pixel calibration saved: {scale:+.6g} px/right press"
            return
        axes = {"X": feedback.dx}
        if self.config.quad_relay:
            axes["Y"] = feedback.dy
        target = "XY" if self.config.quad_relay else "X"
        largest = max(axes, key=lambda axis: abs(axes[axis]))
        if abs(axes[largest]) <= self.context.tolerance_px:
            self._alignment_search = None
            if not feedback.usable:
                # Large Y may allow coarse X alignment, but it must never let
                # an uncertain near-zero X estimate confirm completion.
                self._confirmations = 0
                self.message = f"{target} is within tolerance; waiting for a valid match to confirm alignment"
                return
            self._confirmations += 1
            self.message = f"Confirming {target} alignment ({self._confirmations}/3)"
            if self._confirmations >= 3:
                self._operation = None
                self._alignment_search = None
                self.phase = "ready"
                self.message = f"{target} aligned"
            return
        self._confirmations = 0
        if self._corrections >= self.MAX_CORRECTIONS:
            self.cancel(f"{target} alignment reached its 100 correction limit")
            return
        search = self._alignment_search
        if search is None or abs(axes[search.axis]) <= self.context.tolerance_px:
            search = self._alignment_search = AxisSearch(largest, self.context.tolerance_px)
        try:
            direction = search.next_direction(axes[search.axis])
        except ValueError as exc:
            self.cancel(str(exc))
            return
        self._corrections += 1
        self._stage = "moving"
        quality = " · uncertain match at large offset" if not feedback.usable else ""
        label = f"Aligning {search.axis}"
        if search.crossed_target:
            label = f"Crossed {search.axis} target · reversing"
        self.message = f"{label} · correction {self._corrections} ({direction.name.lower()}){quality}"
        self._press(direction, now, "align")

    def _read_reference_input(self, now: float) -> None:
        channel = self.config.reference_channel
        if channel is None:
            return
        level = self.inputs4[channel]
        rising = self._reference_level is False and level
        # Always consume the transition, including during cooldown. Holding
        # HIGH must never replay a suppressed edge when the second expires.
        self._reference_level = level
        if rising and (self._last_reference_capture_at is None or
                       now - self._last_reference_capture_at >= REFERENCE_CAPTURE_INTERVAL_S):
            self._last_reference_capture_at = now
            self.reference_capture = ReferenceCaptureRequest(channel, now, self.config_revision)

    def _read_align_input(self) -> None:
        channel = self.config.align_channel
        if channel is None:
            return
        level = self.inputs4[channel]
        rising = self._align_level is False and level
        # Consume busy/blocked presses too, so a held input cannot restart
        # an operation after completion, Stop, or a failed readiness check.
        self._align_level = level
        if rising and not (self._operation or self._pending_operation or self._move or self._stopping):
            self.request_align()

    def tick(self, now: float, hardware_state: JoystickState | None = None,
             inputs: tuple[bool, bool, bool, bool] | None = None,
             feedback: MotionFeedback | None = None) -> tuple[MotionAction, ...]:
        if hardware_state is not None:
            self.state = hardware_state
        if feedback is not None:
            self._feedback = feedback
        if inputs is not None:
            self.inputs4 = tuple(bool(value) for value in inputs)
            debounced = list(self._debounced)
            for channel, raw in enumerate(self.inputs4):
                if raw != self._candidate_inputs[channel]:
                    self._input_since[channel] = now
                # Release each contact immediately, then require the complete
                # debounce interval for a subsequent activation, including a
                # bounce back to its former level.
                if raw != self.config.active_high or now - self._input_since[channel] >= INPUT_DEBOUNCE_S:
                    debounced[channel] = raw
            self._candidate_inputs = self.inputs4
            self._debounced = tuple(debounced)
        if not self.connected:
            self._actions.clear()
            return ()
        if inputs is not None:
            self._read_reference_input(now)
            self._read_align_input()
        if self.state.fault and (self.armed or self._move or self._operation):
            self.cancel(f"Motion fault: {self.state.fault}", disarm=True)
        raw = self._logical_inputs(raw=True)
        pressed = [direction for direction, active, level in zip(KEY_DIRECTIONS, self._logical_inputs(), raw)
                   if active and level]
        if (sum(raw) > 1 or self._saw_conflict) and not self._neutral_required:
            self.cancel("Opposing directions: release both controls to continue", disarm=False)
        self._saw_conflict = False
        if self._operation and (any(raw) or self._manual_activity):
            self.cancel("Manual input cancelled camera-feedback motion; release both controls", disarm=False)
        self._manual_activity = False
        if self._stopping:
            if not self.state.busy:
                self._move = None
                self._stopping = False
                self._actions.append(MotionAction("disable"))
                self.phase = "ready" if self.armed else "connected"
            else:
                actions, self._actions = tuple(self._actions), []
                return actions
        if self._move:
            purpose, started, timeout = self._move
            if now - started > timeout:
                self.cancel("Joystick nudge timed out", disarm=True)
            elif now > started and not self.state.busy:
                self._move = None
                if purpose == "jog":
                    self.phase = "ready"
                elif self._operation:
                    self._stage = "postmove"
                    self.message = "Waiting for a fresh camera frame after the nudge"
                    self._freshness(now)
        if self._pending_operation:
            kind, presses = self._pending_operation
            self._pending_operation = None
            self._start_operation(kind, presses, now)
        self._process_operation(now)
        if not any(raw):
            self._neutral_required = False
            if self.armed and self.phase == "waiting":
                self.phase, self.message = "ready", "Ready · hold to nudge one move at a time"
        direction = pressed[0] if len(pressed) == 1 else None
        if (self.armed and not self._neutral_required and not self._operation and
                not self._stopping and direction):
            if direction != self._direction:
                self._next_nudge_at = now
            if self._move is None and not self.state.busy and now >= self._next_nudge_at:
                self._press(direction, now, "jog")
                self.message = f"Nudging {direction.name.lower()}"
        if self._operation == "align_direct" and self._move and self.state.alignment_contact_confirmed:
            self._actions.append(MotionAction("move_after_contact"))
        self._direction = direction
        actions, self._actions = tuple(self._actions), []
        return actions

    def snapshot(self) -> MotionStatus:
        feedback = self._feedback
        return MotionStatus(
            connected=self.connected, setup_applied=self.setup_applied, armed=self.armed,
            phase=self.phase, message=self.message, inputs4=self.inputs4,
            corrections=self._corrections,
            dx=feedback.dx if feedback and math.isfinite(feedback.dx) else None,
            config=self.config, config_revision=self.config_revision,
            reference_capture=self.reference_capture, connection_revision=self.connection_revision,
            busy=self.state.busy, usb_access_needed=self.usb_access_needed,
        )


class MotionWorker(threading.Thread):
    """Serialize hardware requests; UI methods only replace data or enqueue commands."""

    def __init__(self, config: MotionConfig | None = None, hardware=None,
                 freshness_callback: Callable[[float], int] | None = None,
                 calibration_worker: MeasurementWorker | None = None, *,
                 joystick: JoystickPort | None = None):
        super().__init__(name="joystick-motion", daemon=True)
        self.controller = MotionController(config)
        if joystick is not None and hardware is not None:
            raise ValueError("Choose either a joystick port or relay hardware")
        self.wired_joystick = None
        if joystick is None:
            hardware = hardware if hardware is not None else TinkerforgeMotionHardware()
            self.wired_joystick = WiredJoystick(hardware, lambda: self.controller.config,
                                               stop_requested=self._movement_stop_requested)
            joystick = self.wired_joystick
        self.joystick = joystick
        # Older injected joystick ports called their one-nudge command pulse.
        self._nudge = joystick.nudge if hasattr(joystick, "nudge") else joystick.pulse
        self.freshness_callback = freshness_callback
        self.calibration_worker = calibration_worker
        self._calibration_request: tuple[float, int] | None = None
        self.output = LatestValue()
        self.stopping = threading.Event()
        self._stop_requested = threading.Event()
        self._lock = threading.Lock()
        self._commands = deque()
        self._context = MotionContext()
        self._keyboard = (False,) * 4
        self._keyboard_conflict = False
        self._keyboard_activity = False
        self._feedback = None
        self._config_revision = 0
        self._connection_revision = 0
        self.output.publish(self.controller.snapshot())

    def _enqueue(self, name, value=None):
        with self._lock:
            self._commands.append((name, value))

    def configure(self, config: MotionConfig) -> int:
        config.validated()
        with self._lock:
            self._config_revision += 1
            revision = self._config_revision
            self._commands.append(("configure", (config, revision)))
            return revision

    def set_context(self, context: MotionContext):
        with self._lock:
            self._context = context

    def set_keyboard(self, left: bool, right: bool, up=False, down=False):
        with self._lock:
            keyboard = bool(left), bool(right), bool(up), bool(down)
            if keyboard != self._keyboard:
                self._keyboard = keyboard
                self._keyboard_conflict |= sum(keyboard) > 1
                self._keyboard_activity |= any(keyboard)

    def _sync_keyboard(self):
        # Input is a current state, never a queue of presses to replay after I/O.
        # Retain only safety signals from transient conflicts/manual overrides.
        with self._lock:
            keyboard = self._keyboard
            conflict, activity = self._keyboard_conflict, self._keyboard_activity
            self._keyboard_conflict = self._keyboard_activity = False
        self.controller.set_keyboard(*keyboard, conflict=conflict, activity=activity)

    def submit_feedback(self, pair):
        feedback = pair if isinstance(pair, MotionFeedback) else MotionFeedback.from_pair(pair)
        with self._lock:
            self._feedback = feedback

    def _sync_feedback(self, feedback):
        controller = self.controller
        if self.calibration_worker is not None and controller._operation == "calibration":
            observation, _ = self.calibration_worker.calibration_output.read()
            if observation is not None and observation.request == self._calibration_request:
                frame, result = observation.frame, observation.result
                # Keep context invalidation and acquisition provenance while
                # using displacement measured against the pre-move live image.
                feedback = MotionFeedback(
                    frame.frame_id, controller.context.reference_id, controller.context.config_id,
                    result.dx, result.usable, frame.received_at, frame.read_started_at,
                    frame.freshness_generation, dy=result.dy, validity=result.validity)
            else:
                feedback = None
        elif self._calibration_request is not None:
            self.calibration_worker.stop_calibration()
            self._calibration_request = None
        controller.submit_feedback(feedback)

    def _request_connection(self, name):
        with self._lock:
            self._connection_revision += 1
            self._commands.append((name, self._connection_revision))
            return self._connection_revision

    def connect_hardware(self): return self._request_connection("connect")
    def disconnect_hardware(self): return self._request_connection("disconnect")
    def apply_setup(self): self._enqueue("setup")
    def arm(self): self._enqueue("arm")
    def stop(self):
        with self._lock:
            self._stop_requested.set()
            self._commands.append(("stop", None))
    def align(self): self._enqueue("align")
    def align_direct(self): self._enqueue("align_direct")
    def calibrate(self, presses: int):
        if type(presses) is not int or not 1 <= abs(presses) <= 1_000_000:
            raise ValueError("Calibration needs a nonzero signed number of joystick presses (±1000000)")
        self._enqueue("calibrate", presses)
    def stop_worker(self):
        self._stop_requested.set()
        self.stopping.set()

    def _movement_stop_requested(self):
        return self._stop_requested.is_set() or self.stopping.is_set()

    def _execute(self, action):
        if action.kind in {"pulse", "move_after_contact", "enable", "freshness"} and self._movement_stop_requested():
            self.controller.cancel("Motion stopped; release both directions before the next press", disarm=False)
            return
        if action.kind == "pulse":
            if self.wired_joystick is None:
                # An injected port may have disabled its contacts after a Stop.
                self.joystick.enable()
                if self._movement_stop_requested():
                    self.joystick.stop()
                    return
            if action.alignment and isinstance(self.joystick, WiredJoystick):
                self.joystick.nudge(action.direction, alignment=True)
            else:
                self._nudge(action.direction)
        elif action.kind == "move_after_contact":
            self.joystick.move_after_contact()
        elif action.kind == "stop":
            self.joystick.stop()
        elif action.kind == "disable":
            self.joystick.disable()
        elif action.kind == "enable":
            self.joystick.enable()
        elif action.kind == "freshness":
            if self.freshness_callback is None:
                self.controller.cancel("A live camera freshness barrier is required")
            else:
                generation = self.freshness_callback(action.after)
                self.controller.acknowledge_freshness(generation)
                if (self.calibration_worker is not None and self.controller._operation == "calibration"
                        and self.controller._stage == "baseline" and generation > 0):
                    self._calibration_request = action.after, generation
                    self.calibration_worker.start_calibration(action.after, generation)

    def _command(self, name, value):
        controller = self.controller
        if name == "configure":
            config, revision = value
            connection_changed = any(getattr(config, key) != getattr(controller.config, key) for key in
                                     ("host", "port", "digital_uid", "output_uid", "quad_relay"))
            if connection_changed and controller.connected:
                self.joystick.shutdown()
                controller.on_disconnected("Motion connection settings changed; reconnect")
            controller.configure(config, revision)
        elif name == "connect":
            controller.connection_revision = value
            if controller.connected:
                controller.message = "Motion hardware is already connected"
                return
            self.joystick.connect()
            controller.on_connected(self.joystick.read_state())
            # Prepare the axis for use after either startup or a manual connection.
            # Require neutral once so controls held during connection cannot
            # start a move or replay a tap received while hardware was opening.
            self._sync_keyboard()
            self._command("arm", True)
        elif name == "disconnect":
            controller.connection_revision = value
            if controller.connected:
                self.joystick.shutdown()
            controller.on_disconnected()
        elif name == "setup":
            if not controller.connected:
                controller.message = "Connect motion hardware before preparing joystick outputs"
            else:
                controller.cancel("Preparing joystick outputs", disarm=True)
                controller.state = self.joystick.read_state()
                if controller.state.busy:
                    controller.message = "Wait for the joystick outputs to release"
                else:
                    self.joystick.stop()
                    self.joystick.disable()
                    self.joystick.setup()
                    controller.on_setup_applied()
        elif name == "arm":
            if controller.connected:
                if not controller.setup_applied:
                    self._command("setup", None)
                    if not controller.setup_applied:
                        return
                controller.state = self.joystick.read_state()
                controller.inputs4 = self.joystick.read_inputs()
            controller.arm(wait_for_neutral=bool(value))
        elif name == "stop":
            controller.cancel("Motion stopped; release both directions before the next press", disarm=False)
            with self._lock:
                if not any(command[0] == "stop" for command in self._commands):
                    self._stop_requested.clear()
        elif name == "align": controller.request_align()
        elif name == "align_direct": controller.request_align_direct()
        elif name == "calibrate": controller.request_calibration(value)

    def run(self):
        try:
            while not self.stopping.is_set():
                with self._lock:
                    commands, self._commands = self._commands, deque()
                    context, feedback = self._context, self._feedback
                try:
                    self.controller.set_context(context)
                    self._sync_keyboard()
                    for name, value in commands:
                        self._command(name, value)
                    self._sync_feedback(feedback)
                    if self.controller.connected:
                        inputs = self.joystick.read_inputs()
                        state = self.joystick.read_state()
                        # SDK reads can take a network timeout. Observe key
                        # releases and Stop requests received during that wait
                        # before scheduling another finite move.
                        with self._lock:
                            context, feedback = self._context, self._feedback
                        self.controller.set_context(context)
                        self._sync_feedback(feedback)
                        self._sync_keyboard()
                        if self._stop_requested.is_set():
                            self.controller.cancel("Motion stopped; release both directions before the next press", disarm=False)
                        for action in self.controller.tick(time.monotonic(), state, inputs):
                            self._execute(action)
                except Exception as exc:
                    shutdown_error = None
                    try:
                        self.joystick.shutdown()
                    except Exception as shutdown_exc:
                        shutdown_error = shutdown_exc
                    detail = f"Motion error: {exc}"
                    if shutdown_error is not None:
                        detail += f"; hardware shutdown incomplete: {shutdown_error}"
                    self.controller.on_disconnected(detail, usb_access_needed=isinstance(exc, MasterBrickAccessError))
                status = self.controller.snapshot()
                if self.wired_joystick and status.connected:
                    status = replace(status, inputs4=self.wired_joystick.raw_inputs)
                self.output.publish(status)
                self.stopping.wait(.02)
        finally:
            if self._calibration_request is not None:
                self.calibration_worker.stop_calibration()
            try:
                self.joystick.shutdown()
                self.controller.on_disconnected()
            except Exception as exc:
                self.controller.on_disconnected(f"Hardware shutdown incomplete: {exc}")
            self.output.publish(self.controller.snapshot())
