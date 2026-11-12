"""Finite single-axis motion and camera-feedback alignment, independent of Qt.

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

from .motion_hardware import HardwareState, TinkerforgeMotionHardware
from .workers import LatestValue, MeasurementWorker


REFERENCE_CAPTURE_INTERVAL_S = 1.0


@dataclass(frozen=True)
class MotionConfig:
    host: str = "localhost"
    port: int = 4223
    stepper_uid: str = "2eoQ"
    digital_uid: str = "27s4"
    left_channel: int = 2
    right_channel: int = 1
    reference_channel: int | None = None
    active_high: bool = True
    invert_direction: bool = True
    motor_current_ma: int | None = 800
    step_resolution: int = 8
    interpolation: bool = True
    jog_unit: str = "driversteps"
    jog_distance: float = 1.0
    jog_speed: float = 200.0
    repeat_hz: float = 2.0
    pixels_per_step: float | None = None
    calibration_source_id: str = ""
    calibration_shape: tuple[int, int] | None = None

    def validated(self) -> MotionConfig:
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("Brick Daemon host is required")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("Brick Daemon port must be between 1 and 65535")
        if not all(isinstance(v, str) and v.strip() for v in (self.stepper_uid, self.digital_uid)):
            raise ValueError("Both motion Bricklet UIDs are required")
        if self.stepper_uid == self.digital_uid:
            raise ValueError("Motion Bricklets must have different UIDs")
        if any(type(v) is not int or v not in range(4) for v in (self.left_channel, self.right_channel)):
            raise ValueError("Joystick channels must be between 0 and 3")
        if self.left_channel == self.right_channel:
            raise ValueError("Left and right must use different digital inputs")
        if self.reference_channel is not None and (type(self.reference_channel) is not int or self.reference_channel not in range(4)):
            raise ValueError("Reference capture channel must be disabled or between 0 and 3")
        if any(type(v) is not bool for v in (self.active_high, self.invert_direction, self.interpolation)):
            raise ValueError("Input polarity, direction and interpolation must be booleans")
        if self.motor_current_ma is not None and (type(self.motor_current_ma) is not int or not 360 <= self.motor_current_ma <= 1640):
            raise ValueError("Enter the motor's permitted current, between 360 and 1640 mA")
        if type(self.step_resolution) is not int or self.step_resolution not in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            raise ValueError("Step resolution must be a supported microstep denominator")
        if self.jog_unit not in {"driversteps", "pixels"}:
            raise ValueError("Unknown jog distance unit")
        for name in ("jog_distance", "jog_speed", "repeat_hz"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name.replace('_', ' ').capitalize()} must be positive and finite")
        if not .1 <= self.repeat_hz <= 10:
            raise ValueError("Hold repeat frequency must be between 0.1 and 10 Hz")
        if self.pixels_per_step is not None and (isinstance(self.pixels_per_step, bool) or
                not isinstance(self.pixels_per_step, (int, float)) or
                not math.isfinite(self.pixels_per_step) or self.pixels_per_step == 0):
            raise ValueError("Pixel calibration scale must be finite and nonzero")
        if not isinstance(self.calibration_source_id, str):
            raise ValueError("Calibration source identifier must be text")
        if self.calibration_shape is not None and (len(self.calibration_shape) != 2 or any(type(v) is not int or v <= 0 for v in self.calibration_shape)):
            raise ValueError("Calibration image geometry must contain two positive integers")
        self.distance_steps()
        self.speed_steps()
        return self

    def _factor(self) -> float:
        if self.jog_unit == "pixels":
            if self.pixels_per_step is None:
                raise ValueError("Calibrate image pixels per driver step before using pixels")
            return 1 / abs(self.pixels_per_step)
        return 1.0

    def distance_steps(self, value: float | None = None) -> int:
        value = self.jog_distance if value is None else value
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Nudge distance must be positive and finite")
        requested_steps = value * self._factor()
        if not math.isfinite(requested_steps) or requested_steps < 1:
            raise ValueError("Nudge distance must resolve to at least one driver step and fit the motor range")
        steps = round(requested_steps)
        if not 1 <= steps <= 2_147_483_647:
            raise ValueError("Nudge distance must resolve to at least one driver step and fit the motor range")
        return steps

    def speed_steps(self) -> int:
        requested_speed = self.jog_speed * self._factor()
        if not math.isfinite(requested_speed) or requested_speed < 1:
            raise ValueError("Nudge speed must resolve to between 1 and 65535 driver steps/s")
        speed = round(requested_speed)
        if not 1 <= speed <= 65_535:
            raise ValueError("Nudge speed must resolve to between 1 and 65535 driver steps/s")
        return speed

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> MotionConfig:
        if not isinstance(value, dict):
            raise ValueError("Motion settings must be an object")
        names = {f.name for f in fields(cls)}
        kwargs = {key: item for key, item in value.items() if key in names}
        if kwargs.get("jog_unit") == "mm":
            # Older profiles stored physical units. Preserve their effective
            # motor move while retiring the millimetre setting.
            scale = value.get("steps_per_mm")
            if (isinstance(scale, bool) or not isinstance(scale, (int, float)) or
                    not math.isfinite(scale) or scale <= 0):
                raise ValueError("Legacy millimetre settings need positive driver steps per millimetre")
            for name in ("jog_distance", "jog_speed"):
                amount = value.get(name, getattr(cls, name))
                if (isinstance(amount, bool) or not isinstance(amount, (int, float)) or
                        not math.isfinite(amount) or amount <= 0):
                    raise ValueError(f"Legacy {name.replace('_', ' ')} must be positive and finite")
            kwargs["jog_unit"] = "driversteps"
            kwargs["jog_distance"] = value.get("jog_distance", 1.0) * scale
            kwargs["jog_speed"] = value.get("jog_speed", 200.0) * scale
        if kwargs.get("motor_current_ma") is None:
            kwargs["motor_current_ma"] = 800
        if kwargs.get("calibration_shape") is not None:
            kwargs["calibration_shape"] = tuple(kwargs["calibration_shape"])
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
    steps: int = 0
    velocity: int = 0
    after: float = 0.0


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
    position: int = 0
    velocity: int = 0
    remaining: int = 0
    motor_current_ma: int = 0
    step_resolution: int = 8
    interpolation: bool = True
    pixels_per_step: float | None = None
    calibration_source_id: str = ""
    calibration_shape: tuple[int, int] | None = None
    corrections: int = 0
    dx: float | None = None
    config: MotionConfig = MotionConfig()
    config_revision: int = 0
    reference_capture: ReferenceCaptureRequest | None = None


class MotionController:
    """A tick-driven state machine producing finite hardware actions."""

    INPUT_DEBOUNCE_S = .03
    SETTLE_S = .20
    MAX_OPERATION_S = 60.0
    MAX_CORRECTIONS = 100
    FEEDBACK_RETRY_S = 3.0
    UNCERTAIN_ALIGNMENT_MIN_OFFSET_PX = 10.0

    def __init__(self, config: MotionConfig | None = None):
        self.config = (config or MotionConfig()).validated()
        self.config_revision = 0
        self.context = MotionContext()
        self.connected = False
        self.setup_applied = False
        self.armed = False
        self.phase = "disconnected"
        self.message = "Connect motion hardware to begin"
        self.state = HardwareState()
        self.inputs4 = (False, False, False, False)
        self.reference_capture: ReferenceCaptureRequest | None = None
        self._reference_level: bool | None = None
        self._last_reference_capture_at: float | None = None
        self.keyboard = (False, False)
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
        self._calibration_start = (0.0, 0)
        self._calibration_end_position = 0
        self._calibration_steps = 0
        self._corrections = 0
        self._confirmations = 0
        self._previous_error: float | None = None
        self._no_progress = 0
        self._stopping = False

    def configure(self, config: MotionConfig, revision: int = 0) -> None:
        config = config.validated()
        old = self.config
        if self._operation or self._pending_operation:
            self.cancel("Motion settings changed", disarm=False)
        if config.step_resolution != old.step_resolution:
            config = self._without_scales(config)
        if config.pixels_per_step is not None and not config.calibration_source_id and self.context.shape is not None:
            config = replace(config, calibration_source_id=self.context.source_id, calibration_shape=self.context.shape)
        setup_changed = any(getattr(old, name) != getattr(config, name) for name in
                            ("motor_current_ma", "step_resolution", "interpolation"))
        controls_changed = any(getattr(old, name) != getattr(config, name) for name in
                               ("left_channel", "right_channel", "active_high", "invert_direction"))
        if config.reference_channel != old.reference_channel:
            self._reference_level = None
            self.reference_capture = None
        self.config = config
        self.config_revision = revision
        if setup_changed:
            self.setup_applied = False
            self.cancel("Motor settings changed; click Enable motor to apply them", disarm=True)
        elif controls_changed:
            self.cancel("Joystick settings changed; release both directions", disarm=False)
        self._invalidate_camera_scale()

    @staticmethod
    def _without_scales(config: MotionConfig) -> MotionConfig:
        try:
            distance, speed = config.distance_steps(), config.speed_steps()
        except ValueError:
            distance, speed = 1, 200
        loses_unit = config.jog_unit == "pixels"
        return replace(config, pixels_per_step=None, calibration_source_id="", calibration_shape=None,
                       jog_unit="driversteps" if loses_unit else config.jog_unit,
                       jog_distance=float(distance) if loses_unit else config.jog_distance,
                       jog_speed=float(speed) if loses_unit else config.jog_speed)

    def _invalidate_camera_scale(self):
        if self.config.pixels_per_step is None or not self.context.source_id or self.context.shape is None:
            return
        if (self.config.calibration_source_id != self.context.source_id or
                self.config.calibration_shape != self.context.shape):
            self.config = self._without_scales(self.config)
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

    def set_keyboard(self, left: bool, right: bool, *, conflict=False, activity=False) -> None:
        self.keyboard = bool(left), bool(right)
        after = self._logical_inputs(raw=True)
        if conflict or (after[0] and after[1]):
            self._saw_conflict = True
        if (activity or any(after)) and self._operation:
            self._manual_activity = True

    def submit_feedback(self, feedback: MotionFeedback | None) -> None:
        self._feedback = feedback

    def on_connected(self, state: HardwareState) -> None:
        self.connected = True
        self._reference_level = None
        self.reference_capture = None
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
        self.message = "Motor is disabled. Click Enable motor to apply settings and begin"

    def on_disconnected(self, message="Motion hardware disconnected") -> None:
        self.connected = self.setup_applied = self.armed = False
        self._reference_level = None
        self.reference_capture = None
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
        self.phase, self.message = "connected", "Motor settings applied. Click Enable motor with joystick and arrow keys released"

    def arm(self, *, wait_for_neutral: bool = False):
        if not self.connected or not self.setup_applied:
            self.message = "Connect hardware and apply motor settings before enabling movement"
            return
        if self.state.fault:
            self.message = self.state.fault
            return
        if any(self._logical_inputs(raw=True)):
            held = []
            for name, channel in (("Left", self.config.left_channel), ("Right", self.config.right_channel)):
                if self.inputs4[channel] == self.config.active_high:
                    level = "HIGH" if self.inputs4[channel] else "LOW"
                    held.append(f"{name} joystick input (channel {channel}: {level})")
            held.extend(f"{name} arrow key" for name, pressed in zip(("Left", "Right"), self.keyboard) if pressed)
            self.message = "Release controls before moving: " + "; ".join(held)
            if not self.config.active_high and any(
                    not self.inputs4[channel] for channel in (self.config.left_channel, self.config.right_channel)):
                self.message += ". If joystick inputs are LOW when released, check Active high"
            if not wait_for_neutral:
                return
        if self._move or self._stopping or self.state.velocity or self.state.remaining:
            self.message = "Wait for the motor to stop, then click Enable motor"
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
        self._pending_operation = ("align", 0)

    def request_calibration(self, steps: int):
        if type(steps) is not int or steps == 0 or abs(steps) > 2_147_483_647:
            raise ValueError("Calibration needs a nonzero signed number of driver steps")
        self._pending_operation = ("calibration", steps)

    def cancel(self, message="Motion stopped; release both directions before the next press", *, disarm=False):
        # Recoverable operation failures stop motion without requiring another
        # Enable click. Hardware faults and setup changes explicitly disarm.
        self._operation = self._pending_operation = None
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
        inputs = self.inputs4 if raw else self._debounced
        left = inputs[self.config.left_channel] == self.config.active_high
        right = inputs[self.config.right_channel] == self.config.active_high
        return left or self.keyboard[0], right or self.keyboard[1]

    def _freshness(self, now):
        self._fresh_cutoff = now + max(self.SETTLE_S, self.context.settling_s)
        self._required_generation = None
        self._last_consumed_frame = None
        self._actions.append(MotionAction("freshness", after=self._fresh_cutoff))

    def _start_move(self, steps, now, purpose):
        velocity = self.config.speed_steps()
        self._move = (purpose, self.state.position, steps, now, max(5.0, abs(steps) / velocity * 3 + 1))
        self._actions.append(MotionAction("move", steps=steps, velocity=velocity))
        self.phase = {"jog": "jogging", "align": "aligning", "calibration": "calibrating"}[purpose]

    def _start_operation(self, kind, steps, now):
        if not self.armed or not self.setup_applied:
            self.message = "Click Enable motor before using camera feedback"
            return
        if self._move or self.state.velocity or self.state.remaining or self._stopping:
            self.message = "Wait for the current nudge to finish"
            return
        if any(self._logical_inputs(raw=True)):
            self.message = "Release joystick and arrow keys before using camera feedback"
            return
        if not self.context.live_camera_ready:
            self.message = "A running, unpaused IDS or Basler camera is required"
            return
        if kind == "align" and not self.context.camera_ready:
            self.message = "A running, unpaused IDS or Basler camera and a reference are required"
            return
        if kind == "align" and self.config.pixels_per_step is None:
            self.message = "Calibrate image pixels per driver step before aligning"
            return
        self._operation, self._stage = kind, "baseline"
        self._op_started = now
        self._feedback_retry_started_at = None
        self._calibration_steps = steps
        self._corrections = self._confirmations = self._no_progress = 0
        self._previous_error = None
        self._feedback = None
        self._neutral_required = True
        self.phase = "aligning" if kind == "align" else "calibrating"
        self.message = "Waiting for a fresh, settled camera measurement"
        self._freshness(now)

    def _feedback_usable(self, feedback):
        if not all(math.isfinite(value) for value in (feedback.dx, feedback.dy)):
            return False
        if feedback.usable:
            return True
        # Permit uncertain estimates for coarse alignment only. Calibration
        # still needs valid measurements to establish the motor's pixel scale.
        threshold = max(self.UNCERTAIN_ALIGNMENT_MIN_OFFSET_PX, self.context.tolerance_px)
        return (self._operation == "align" and feedback.validity == "uncertain"
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
            if (self._operation == "align" and feedback.validity == "uncertain"
                    and math.isfinite(feedback.dx) and math.isfinite(feedback.dy)):
                self._retry_operation_feedback(now, "Camera match is uncertain at low offset; a valid match is required")
            else:
                print(f"DEBUG: feedback unusable: usable={feedback.usable} validity={feedback.validity} dx={feedback.dx} dy={feedback.dy}")
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
        feedback = self._consume_operation_feedback(now)
        if feedback is None or not self._operation:
            return
        if self._operation == "calibration":
            if self._stage == "baseline":
                self._calibration_start = feedback.dx, self.state.position
                self._stage = "moving"
                self.message = "Calibration test move in progress"
                self._start_move(self._calibration_steps, now, "calibration")
            elif self._stage == "postmove":
                dx0, position0 = self._calibration_start
                delta_steps = self._calibration_end_position - position0
                delta_pixels = feedback.dx - dx0
                if delta_steps == 0 or abs(delta_pixels) < 2:
                    self.cancel("Calibration needs actual motor movement and at least 2 pixels of observed X movement")
                    return
                scale = delta_pixels / delta_steps
                if not math.isfinite(scale) or scale == 0:
                    self.cancel("Pixel calibration did not produce a finite nonzero scale")
                    return
                candidate = replace(self.config, pixels_per_step=scale,
                                    calibration_source_id=self.context.source_id, calibration_shape=self.context.shape)
                # Show the calibrated nudge in pixels, preserving its selected
                # motor distance and speed through the unit change.
                candidate = replace(candidate, jog_unit="pixels",
                                    jog_distance=self.config.distance_steps() * abs(scale),
                                    jog_speed=self.config.speed_steps() * abs(scale))
                self.config = candidate.validated()
                self._operation = None
                self.phase = "ready"
                self.message = (f"Pixel calibration saved: {self.config.jog_distance:.6g} px/nudge "
                                f"({scale:+.6g} px/driver step)")
            return
        error = abs(feedback.dx)
        if error <= self.context.tolerance_px:
            if not feedback.usable:
                # Large Y may allow coarse X alignment, but it must never let
                # an uncertain near-zero X estimate confirm completion.
                self._confirmations = 0
                self.message = "X is within tolerance; waiting for a valid match to confirm alignment"
                return
            self._confirmations += 1
            self.message = f"Confirming X alignment ({self._confirmations}/3)"
            if self._confirmations >= 3:
                self._operation = None
                self.phase = "ready"
                self.message = "X aligned · Y is unchanged by this axis"
            return
        self._confirmations = 0
        if self._previous_error is not None:
            allowance = max(.25, self.context.tolerance_px * .1)
            if error > self._previous_error + allowance:
                self.cancel("X error increased; check motor direction and pixel calibration")
                return
            self._no_progress = self._no_progress + 1 if self._previous_error - error < max(.05, self.context.tolerance_px * .1) else 0
            if self._no_progress >= 10:
                self.cancel("X alignment stopped after 10 moves without progress")
                return
        if self._corrections >= self.MAX_CORRECTIONS:
            self.cancel("X alignment reached its 100 correction limit")
            return
        limit = self.config.distance_steps()
        steps = round(-feedback.dx / self.config.pixels_per_step)
        steps = max(-limit, min(limit, steps))
        if steps == 0:
            self.cancel("Requested X tolerance is smaller than one driver step can resolve")
            return
        self._previous_error = error
        self._corrections += 1
        self._stage = "moving"
        quality = " · uncertain match at large offset" if not feedback.usable else ""
        self.message = f"Aligning X · correction {self._corrections}{quality}"
        self._start_move(steps, now, "align")

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

    def tick(self, now: float, hardware_state: HardwareState | None = None,
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
                if raw != self.config.active_high or now - self._input_since[channel] >= self.INPUT_DEBOUNCE_S:
                    debounced[channel] = raw
            self._candidate_inputs = self.inputs4
            self._debounced = tuple(debounced)
        if not self.connected:
            self._actions.clear()
            return ()
        if inputs is not None:
            self._read_reference_input(now)
        if self.state.fault and (self.armed or self._move or self._operation):
            self.cancel(f"Motor fault: {self.state.fault}", disarm=True)
        if self.setup_applied and self.state.step_resolution != self.config.step_resolution:
            self.config = self._without_scales(self.config)
            self.setup_applied = False
            self.cancel("Device microstep resolution changed; reapply setup and recalibrate", disarm=True)
        left, right = self._logical_inputs()
        # Raw conflicting contacts stop immediately; debounce only delays starts.
        raw_left, raw_right = self._logical_inputs(raw=True)
        # Releases are immediate. A debounced former direction must never
        # survive a physical reversal while the new contact is settling.
        left, right = left and raw_left, right and raw_right
        if (raw_left and raw_right or self._saw_conflict) and not self._neutral_required:
            self.cancel("Opposing directions: release both controls to continue", disarm=False)
        self._saw_conflict = False
        if self._operation and (raw_left or raw_right or self._manual_activity):
            self.cancel("Manual input cancelled camera-feedback motion; release both controls", disarm=False)
        self._manual_activity = False
        if self._stopping:
            if self.state.velocity == 0 and self.state.remaining == 0:
                self._move = None
                self._stopping = False
                self._actions.append(MotionAction("disable"))
                self.phase = "ready" if self.armed else "connected"
            else:
                actions, self._actions = tuple(self._actions), []
                return actions
        if self._move:
            purpose, start_position, steps, started, timeout = self._move
            if now - started > timeout:
                self.cancel("Motor move timed out", disarm=True)
            elif now > started and self.state.velocity == 0 and self.state.remaining == 0:
                self._move = None
                if purpose == "jog":
                    self.phase = "ready"
                elif self._operation:
                    self._calibration_end_position = self.state.position
                    self._stage = "postmove"
                    self.message = "Waiting for a fresh camera frame after motor stop"
                    self._freshness(now)
        if self._pending_operation:
            kind, steps = self._pending_operation
            self._pending_operation = None
            self._start_operation(kind, steps, now)
        self._process_operation(now)
        if not raw_left and not raw_right:
            self._neutral_required = False
            if self.armed and self.phase == "waiting":
                self.phase, self.message = "ready", "Ready · hold to nudge one move at a time"
        direction = -1 if left and not right else 1 if right and not left else 0
        if not raw_left and not raw_right:
            direction = 0
        if (self.armed and not self._neutral_required and not self._operation and
                not self._stopping and direction):
            if direction != self._direction:
                self._next_nudge_at = now
            if self._move is None and now >= self._next_nudge_at:
                sign = -direction if self.config.invert_direction else direction
                self._start_move(sign * self.config.distance_steps(), now, "jog")
                self._next_nudge_at = now + 1 / self.config.repeat_hz
                self.message = "Nudging left" if direction < 0 else "Nudging right"
        self._direction = direction
        actions, self._actions = tuple(self._actions), []
        return actions

    def snapshot(self) -> MotionStatus:
        return MotionStatus(self.connected, self.setup_applied, self.armed, self.phase, self.message,
                            self.inputs4, self.state.position, self.state.velocity, self.state.remaining,
                            self.state.motor_current_ma, self.state.step_resolution, self.state.interpolation,
                            self.config.pixels_per_step, self.config.calibration_source_id,
                            self.config.calibration_shape, self._corrections,
                            self._feedback.dx if self._feedback and math.isfinite(self._feedback.dx) else None,
                            self.config, self.config_revision, self.reference_capture)


class MotionWorker(threading.Thread):
    """Serialize hardware requests; UI methods only replace data or enqueue commands."""

    def __init__(self, config: MotionConfig | None = None, hardware=None,
                 freshness_callback: Callable[[float], int] | None = None,
                 calibration_worker: MeasurementWorker | None = None):
        super().__init__(name="single-axis-motion", daemon=True)
        self.controller = MotionController(config)
        self.hardware = hardware if hardware is not None else TinkerforgeMotionHardware()
        self.freshness_callback = freshness_callback
        self.calibration_worker = calibration_worker
        self._calibration_request: tuple[float, int] | None = None
        self.output = LatestValue()
        self.stopping = threading.Event()
        self._stop_requested = threading.Event()
        self._lock = threading.Lock()
        self._commands = deque()
        self._context = MotionContext()
        self._keyboard = (False, False)
        self._keyboard_conflict = False
        self._keyboard_activity = False
        self._feedback = None
        self._config_revision = 0
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

    def set_keyboard(self, left: bool, right: bool):
        with self._lock:
            keyboard = bool(left), bool(right)
            if keyboard != self._keyboard:
                self._keyboard = keyboard
                self._keyboard_conflict |= all(keyboard)
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

    def connect_hardware(self): self._enqueue("connect")
    def disconnect_hardware(self): self._enqueue("disconnect")
    def apply_setup(self): self._enqueue("setup")
    def arm(self): self._enqueue("arm")
    def stop(self):
        with self._lock:
            self._stop_requested.set()
            self._commands.append(("stop", None))
    def align(self): self._enqueue("align")
    def calibrate(self, steps: int):
        if type(steps) is not int or steps == 0 or abs(steps) > 2_147_483_647:
            raise ValueError("Calibration needs a nonzero signed number of driver steps")
        self._enqueue("calibrate", steps)
    def stop_worker(self):
        self._stop_requested.set()
        self.stopping.set()

    def _execute(self, action):
        if action.kind in {"move", "enable", "freshness"} and (self._stop_requested.is_set() or self.stopping.is_set()):
            self.controller.cancel("Motion stopped; release both directions before the next press", disarm=False)
            return
        if action.kind == "move":
            self.hardware.enable()
            if self._stop_requested.is_set() or self.stopping.is_set():
                self.controller.cancel("Motion stopped; release both directions before the next press", disarm=False)
                self.hardware.stop()
                return
            self.hardware.move_steps(action.steps, action.velocity)
        elif action.kind == "stop":
            self.hardware.stop()
        elif action.kind == "disable":
            self.hardware.disable()
        elif action.kind == "enable":
            self.hardware.enable()
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
                                     ("host", "port", "stepper_uid", "digital_uid"))
            if connection_changed and controller.connected:
                self.hardware.shutdown()
                controller.on_disconnected("Motion connection settings changed; reconnect")
            controller.configure(config, revision)
        elif name == "connect":
            if controller.connected:
                controller.message = "Motion hardware is already connected"
                return
            config = controller.config
            self.hardware.connect(config.host, config.port, config.stepper_uid, config.digital_uid)
            controller.on_connected(self.hardware.read_state())
            # Connection is the user's request to prepare this axis for use.
            # Require neutral once so controls held during connection cannot
            # start a move or replay a tap received while hardware was opening.
            self._sync_keyboard()
            self._command("arm", True)
        elif name == "disconnect":
            if controller.connected:
                self.hardware.shutdown()
            controller.on_disconnected()
        elif name == "setup":
            if not controller.connected:
                controller.message = "Connect motion hardware before applying motor setup"
            else:
                controller.cancel("Applying motor setup", disarm=True)
                controller.state = self.hardware.read_state()
                if controller.state.velocity or controller.state.remaining:
                    controller.message = "Stop the motor before applying setup"
                elif controller.config.motor_current_ma is None:
                    controller.message = "Enter the motor's permitted current before applying setup"
                else:
                    config = controller.config
                    self.hardware.stop()
                    self.hardware.disable()
                    self.hardware.configure(config.motor_current_ma, config.step_resolution, config.interpolation)
                    controller.on_setup_applied()
        elif name == "arm":
            if controller.connected:
                if not controller.setup_applied:
                    self._command("setup", None)
                    if not controller.setup_applied:
                        return
                controller.state = self.hardware.read_state()
                controller.inputs4 = self.hardware.read_inputs()
            controller.arm(wait_for_neutral=bool(value))
        elif name == "stop":
            controller.cancel("Motion stopped; release both directions before the next press", disarm=False)
            with self._lock:
                if not any(command[0] == "stop" for command in self._commands):
                    self._stop_requested.clear()
        elif name == "align": controller.request_align()
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
                        state = self.hardware.read_state()
                        inputs = self.hardware.read_inputs()
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
                        self.hardware.shutdown()
                    except Exception as shutdown_exc:
                        shutdown_error = shutdown_exc
                    detail = f"Motion error: {exc}"
                    if shutdown_error is not None:
                        detail += f"; motor shutdown incomplete: {shutdown_error}"
                    self.controller.on_disconnected(detail)
                self.output.publish(self.controller.snapshot())
                self.stopping.wait(.02)
        finally:
            if self._calibration_request is not None:
                self.calibration_worker.stop_calibration()
            try:
                self.hardware.shutdown()
                self.controller.on_disconnected()
            except Exception as exc:
                self.controller.on_disconnected(f"Motor shutdown incomplete: {exc}")
            self.output.publish(self.controller.snapshot())
