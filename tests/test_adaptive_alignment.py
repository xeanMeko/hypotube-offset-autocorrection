"""Session calibration and alignment against deterministic live camera motion."""
from dataclasses import replace
import math

import pytest

from measure_shift.motion import MotionConfig, MotionContext, MotionController, MotionFeedback
from measure_shift.motion_hardware import HardwareState


CAMERA = MotionContext(source_id="ids:adaptive", source_kind="IDS", shape=(64, 80),
                       reference_id="reference", config_id="match", connected=True)


def calibrated(scale=1, **kwargs):
    return MotionConfig(pixels_per_step=scale, calibration_source_id=CAMERA.source_id,
                        calibration_shape=CAMERA.shape, **kwargs)


class Axis:
    """Keep physical live motion separate from the saved-reference X offset."""

    def __init__(self, *, config=None, error=80, gain=1, context=CAMERA,
                 session_calibrated=None):
        self.controller = MotionController(config or MotionConfig())
        self.context = context
        self.controller.set_context(context)
        self.position = 0
        self.error = error
        self.live_x = 0.0
        self.gain = gain
        self.now = 0.0
        self.generation = 0
        self.frame = 0
        self.cutoff = 0.0
        self.pending_move = None
        self.moves = []
        self.velocities = []
        self.calibration_moves = []
        self.calibration_captures = 0
        self.calibration_origin = None
        self.controller.on_connected(self.hardware())
        self.controller.on_setup_applied()
        self.controller.arm()
        self.controller.tick(self.now)
        if session_calibrated is None:
            session_calibrated = self.controller.config.pixels_per_step is not None
        self.controller._session_calibrated = session_calibrated
        self.start_alignment()

    def hardware(self, **kwargs):
        return HardwareState(position=self.position, motor_current_ma=800,
                             step_resolution=self.controller.config.step_resolution, **kwargs)

    def _actions(self, actions):
        for action in actions:
            if action.kind == "freshness":
                self.cutoff = action.after
                self.generation += 1
                self.controller.acknowledge_freshness(self.generation)
            elif action.kind == "calibration_reference":
                self.calibration_origin = self.live_x
                self.calibration_captures += 1
            elif action.kind == "move":
                assert self.pending_move is None
                self.pending_move = action
                self.moves.append(action.steps)
                self.velocities.append(action.velocity)
                if self.controller._operation == "calibration":
                    self.calibration_moves.append(action.steps)
        return actions

    def start_alignment(self):
        self.controller.request_align()
        self.now += 1.0
        self._actions(self.controller.tick(self.now, self.hardware()))
        assert self.controller._operation in {"align", "calibration"}

    def sample(self, dx=None, *, frame_id=None, usable=True, generation=None,
               started=None, received=None, validity="", dy=0, calibration_usable=True):
        self.now = max(self.now + .01, self.cutoff + .01)
        self.frame += 1
        local_dx = None if self.calibration_origin is None else self.live_x - self.calibration_origin
        value = MotionFeedback(
            str(self.frame) if frame_id is None else frame_id,
            self.context.reference_id, self.context.config_id, self.error if dx is None else dx,
            usable, self.now if received is None else received,
            self.now if started is None else started,
            self.generation if generation is None else generation, dy=dy, validity=validity,
            calibration_dx=local_dx, calibration_usable=calibration_usable)
        return self._actions(self.controller.tick(self.now, self.hardware(), feedback=value))

    def observe(self, values=None, **kwargs):
        assert self.pending_move is None
        for dx in ([self.error] * 3 if values is None else values):
            self.sample(dx, **kwargs)
            if self.pending_move is not None or self.controller._operation is None:
                break
        return self.pending_move

    def finish(self, *, actual_steps=None, pixel_change=None):
        assert self.pending_move is not None
        requested = self.pending_move.steps
        actual_steps = requested if actual_steps is None else actual_steps
        self.position = (self.position + actual_steps + 2**31) % 2**32 - 2**31
        change = self.gain * actual_steps if pixel_change is None else pixel_change
        self.error += change
        self.live_x += change
        self.pending_move = None
        self.now += .02
        self._actions(self.controller.tick(self.now, self.hardware()))

    def run(self, *, noise=(0, 0, 0), max_observations=150):
        for _ in range(max_observations):
            if self.controller._operation is None:
                return
            self.observe([self.error + item for item in noise])
            if self.pending_move is not None:
                self.finish()
        pytest.fail("Session alignment did not finish within the simulation bound")


@pytest.mark.parametrize("value", [0, -1, 1.5, True, 2_147_483_648])
def test_alignment_limit_requires_a_positive_driver_step_integer(value):
    with pytest.raises(ValueError):
        replace(MotionConfig(), align_max_steps=value).validated()


def test_alignment_limit_roundtrips_and_old_settings_default_to_ten():
    assert MotionConfig().align_max_steps == 10
    assert MotionConfig.from_dict({}).align_max_steps == 10
    config = replace(MotionConfig(), align_max_steps=7)
    assert MotionConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize("gain", [1, -1, .2, -.2, 2.5, -2.5])
def test_first_alignment_calibrates_live_motion_then_converges(gain):
    axis = Axis(gain=gain)

    axis.run()

    assert axis.calibration_captures == 1
    assert axis.calibration_moves == [10]
    assert abs(axis.error) <= CAMERA.tolerance_px
    assert "X aligned" in axis.controller.message
    assert axis.controller.config.pixels_per_step == pytest.approx(gain)
    assert axis.controller.config.calibration_source_id == CAMERA.source_id
    assert axis.controller.config.calibration_shape == CAMERA.shape
    assert axis.controller.config.jog_unit == "driversteps"
    assert all(1 <= abs(move) <= 10 for move in axis.moves)


@pytest.mark.parametrize("limit", [1, 3])
def test_small_calibration_moves_accumulate_against_one_live_baseline(limit):
    axis = Axis(config=MotionConfig(align_max_steps=limit), gain=.25, error=16)

    axis.run()

    assert axis.calibration_captures == 1
    assert len(axis.calibration_moves) > 1
    assert set(axis.calibration_moves) == {limit}
    assert axis.controller.config.pixels_per_step == pytest.approx(.25)
    assert "X aligned" in axis.controller.message
    assert all(abs(move) <= limit for move in axis.moves)


def test_established_scale_uses_the_alignment_cap_independent_of_manual_nudge():
    axis = Axis(config=calibrated(jog_distance=1, align_max_steps=7), error=160)

    axis.run()

    assert axis.moves[0] == -7
    assert max(map(abs, axis.moves)) == 7
    assert axis.calibration_captures == 0
    assert axis.controller.config.distance_steps() == 1


def test_large_manual_nudge_never_increases_alignment_cap():
    axis = Axis(config=calibrated(jog_distance=100, align_max_steps=3), error=100)
    axis.run()
    assert max(map(abs, axis.moves)) == 3
    assert axis.controller.config.distance_steps() == 100


def test_subsequent_alignment_reuses_session_calibration_without_a_probe():
    axis = Axis(gain=.5)
    axis.run()
    assert "X aligned" in axis.controller.message
    move_count = len(axis.moves)
    axis.error = 80

    axis.start_alignment()
    axis.observe()

    assert axis.calibration_captures == 1
    assert axis.calibration_moves == [10]
    assert len(axis.moves) == move_count + 1
    assert axis.pending_move.steps == -10
    axis.finish()
    axis.run()
    assert "X aligned" in axis.controller.message


def test_changing_target_reference_preserves_session_calibration():
    axis = Axis(gain=.5)
    axis.run()
    axis.context = replace(CAMERA, reference_id="another-target", config_id="another-match")
    axis.controller.set_context(axis.context)
    axis.error = 40

    axis.start_alignment()
    axis.observe()

    assert axis.calibration_captures == 1
    assert axis.pending_move.steps == -10
    assert axis.controller.config.pixels_per_step == pytest.approx(.5)


@pytest.mark.parametrize("change", ["source", "shape", "step_resolution", "interpolation", "scale"])
def test_camera_or_motor_scale_changes_require_new_session_calibration(change):
    axis = Axis(config=calibrated(), error=.5)
    axis.run()
    assert axis.controller._session_calibrated

    if change == "source":
        axis.controller.set_context(replace(CAMERA, source_id="ids:other"))
    elif change == "shape":
        axis.controller.set_context(replace(CAMERA, shape=(80, 100)))
    else:
        value = {"step_resolution": 16, "interpolation": False, "scale": 2}[change]
        field = "pixels_per_step" if change == "scale" else change
        axis.controller.configure(replace(axis.controller.config, **{field: value}))

    assert not axis.controller._session_calibrated


def test_reconnecting_requires_new_session_calibration():
    axis = Axis(config=calibrated(), error=.5)
    axis.run()

    axis.controller.on_disconnected()
    axis.controller.on_connected(axis.hardware())

    assert not axis.controller._session_calibrated


@pytest.mark.parametrize("usable,dx", [(True, -10_000), (False, math.nan)])
def test_calibration_uses_local_motion_independently_of_target_error(usable, dx):
    axis = Axis(gain=1.5)
    axis.observe([dx], usable=usable, validity="invalid" if not usable else "")
    assert axis.pending_move.steps == 10
    axis.finish()

    axis.observe([dx], usable=usable, validity="invalid" if not usable else "")

    assert axis.controller.config.pixels_per_step == pytest.approx(1.5)
    assert axis.controller._session_calibrated
    assert axis.calibration_captures == 1


def test_calibration_uses_actual_driver_position_for_partial_travel():
    axis = Axis(gain=1.5)
    axis.observe()
    assert axis.pending_move.steps == 10
    axis.finish(actual_steps=4)

    axis.observe()

    assert axis.controller.config.pixels_per_step == pytest.approx(1.5)


def test_calibration_handles_signed_driver_counter_wrap():
    axis = Axis(gain=-2)
    axis.position = 2_147_483_643
    axis.observe()
    axis.finish()
    assert axis.position < 0

    axis.run()

    assert "X aligned" in axis.controller.message
    assert axis.controller.config.pixels_per_step == pytest.approx(-2)


@pytest.mark.parametrize("gain", [0, .001, -.001])
def test_unmeasurable_live_response_stops_at_calibration_travel_budget(gain):
    axis = Axis(gain=gain)

    axis.run()

    assert sum(map(abs, axis.calibration_moves)) == 100
    assert axis.controller.config.pixels_per_step is None
    assert not axis.controller.armed
    assert "calibrat" in axis.controller.message.lower()


def test_zero_driver_travel_cannot_calibrate_from_image_change():
    axis = Axis()
    axis.observe()
    axis.finish(actual_steps=0, pixel_change=3)

    axis.observe()

    assert axis.controller.config.pixels_per_step is None
    assert axis.controller._operation is None
    assert not axis.controller.armed


def test_invalid_local_comparison_cannot_save_a_scale():
    axis = Axis()
    axis.observe()
    axis.finish()

    axis.observe(calibration_usable=False)

    assert axis.controller.config.pixels_per_step is None
    assert axis.controller._operation is None
    assert not axis.controller.armed


def test_automatic_recalibration_preserves_manual_units_and_effective_motion():
    config = calibrated(jog_unit="pixels", jog_distance=7, jog_speed=250)
    axis = Axis(config=config, gain=.8, session_calibrated=False)

    axis.run()

    updated = axis.controller.config
    assert updated.pixels_per_step == pytest.approx(.8)
    assert updated.jog_unit == "pixels"
    assert updated.distance_steps() == config.distance_steps()
    assert updated.speed_steps() == config.speed_steps()
    assert set(axis.velocities) == {250}


def test_settled_corrections_need_three_distinct_fresh_frames():
    axis = Axis(config=calibrated())
    axis.sample(frame_id="old-read", started=0)
    axis.sample(frame_id="old-generation", generation=axis.generation - 1)
    axis.sample(frame_id="one")
    axis.sample(frame_id="one")
    axis.sample(frame_id="two")
    axis.sample(frame_id="one")
    assert not axis.moves

    axis.sample(frame_id="three")

    assert axis.moves == [-10]


def test_median_observations_reject_isolated_camera_outliers():
    axis = Axis(config=calibrated(scale=-1), gain=-1)
    for _ in range(60):
        if abs(axis.error) <= 2:
            break
        axis.observe([axis.error + .02, axis.error + 200, axis.error - .02])
        assert axis.pending_move is not None
        axis.finish()
    axis.run()
    assert abs(axis.error) <= CAMERA.tolerance_px
    assert "X aligned" in axis.controller.message
    assert axis.controller.config.pixels_per_step == -1


def test_aligned_confirmation_requires_three_distinct_valid_frames():
    axis = Axis(config=calibrated(), error=.5)
    axis.sample(frame_id="one")
    axis.sample(frame_id="one")
    assert axis.controller._confirmations == 1
    axis.sample(frame_id="two")
    assert axis.controller._operation == "align"

    axis.sample(frame_id="three")

    assert axis.controller._operation is None
    assert "X aligned" in axis.controller.message
    assert not axis.moves


@pytest.mark.parametrize("usable", [True, False])
def test_no_progress_continues_until_the_existing_correction_limit(usable):
    axis = Axis(config=calibrated(), error=100)
    observation = {} if usable else {"usable": False, "validity": "uncertain"}
    axis.observe(**observation)
    while len(axis.moves) < axis.controller.MAX_CORRECTIONS:
        axis.finish(pixel_change=0)
        axis.observe(**observation)
        assert axis.controller._operation == "align"
        assert axis.pending_move is not None

    axis.finish(pixel_change=0)
    axis.observe(**observation)

    assert axis.controller._operation is None
    assert "correction limit" in axis.controller.message
    assert len(axis.moves) == axis.controller.MAX_CORRECTIONS


@pytest.mark.parametrize("usable", [True, False])
def test_error_growth_allows_another_correction_with_the_same_session_scale(usable):
    axis = Axis(config=calibrated(), error=20)
    observation = {} if usable else {"usable": False, "validity": "uncertain"}
    axis.observe(**observation)
    axis.finish(pixel_change=5)

    axis.observe(**observation)

    assert axis.controller.armed
    assert axis.pending_move.steps == -10
    assert axis.controller.config.pixels_per_step == 1


def test_damped_rounding_takes_one_step_only_when_it_improves_error():
    context = replace(CAMERA, tolerance_px=.01)
    axis = Axis(config=calibrated(scale=2), gain=2, error=1.2, context=context)
    axis.observe()
    assert axis.moves == [-1]
    axis.finish()

    axis.observe()

    assert axis.controller._operation is None
    assert "resolve" in axis.controller.message
    assert axis.moves == [-1]


def test_calibration_and_alignment_share_the_existing_time_limit():
    axis = Axis()
    axis.now = 62
    axis.controller.tick(axis.now, axis.hardware())
    assert axis.controller._operation is None
    assert not axis.controller.armed
    assert "60 second limit" in axis.controller.message


@pytest.mark.parametrize("reason", ["stop", "manual", "fault", "source"])
def test_calibration_stops_for_existing_interruptions(reason):
    axis = Axis()
    axis.observe()
    assert axis.pending_move is not None
    if reason == "stop":
        axis.controller.cancel()
    elif reason == "manual":
        axis.controller.set_keyboard(True, False)
    elif reason == "source":
        axis.controller.set_context(replace(CAMERA, source_id="ids:other"))
    hardware = axis.hardware(fault="Overtemperature" if reason == "fault" else "")

    actions = axis.controller.tick(axis.now + .01, hardware)

    assert any(action.kind == "stop" for action in actions)
    assert axis.controller._operation is None
