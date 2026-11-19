"""Calibration-free motor feedback with slack, unknown direction and scale."""
from dataclasses import replace
import math

import pytest

from measure_shift.joystick import Direction, JoystickState
from measure_shift.motion import MotionConfig
from test_motion import CONTEXT, feedback, moves, ready, state


class DirectAxis:
    def __init__(self, *, error=40, gain=1, nudge=4, slack=0, config=None, direct=True):
        self.controller = ready(config or MotionConfig())
        self.original_config = self.controller.config
        self.error = error
        self.gain = gain
        self.nudge = nudge
        self.slack = slack
        self.remaining_slack = slack
        self.direction = 0
        self.position = 0
        self.now = 1.0
        self.generation = 0
        self.cutoff = 0.0
        self.frame = 0
        self.steps = []
        self.pending = None
        if direct:
            self.controller.request_align_direct()
        else:
            self.controller.request_align()
        self.actions(self.controller.tick(self.now, JoystickState()))

    def actions(self, actions):
        for action in actions:
            if action.kind == "freshness":
                self.generation += 1
                self.cutoff = action.after
                self.controller.acknowledge_freshness(self.generation)
            elif action.kind == "pulse":
                assert self.pending is None
                self.pending = action
                self.steps.append(-action.direction * self.nudge)
        return actions

    def sample(self, dx=None, **kwargs):
        self.now = max(self.now + .01, self.cutoff + .01)
        self.frame += 1
        kwargs.setdefault("generation", self.generation)
        value = feedback(self.frame, self.error if dx is None else dx, self.now, **kwargs)
        return self.actions(self.controller.tick(self.now, JoystickState(), feedback=value))

    def finish(self, actual=None):
        assert self.pending is not None
        steps = self.steps[-1] if actual is None else actual
        self.position = (self.position + steps + 2**31) % 2**32 - 2**31
        direction = 1 if steps > 0 else -1
        if direction != self.direction:
            self.remaining_slack = self.slack
            self.direction = direction
        dead_travel = min(self.remaining_slack, abs(steps))
        self.remaining_slack -= dead_travel
        self.error += direction * (abs(steps) - dead_travel) * self.gain
        self.pending = None
        self.now += .02
        return self.actions(self.controller.tick(self.now, JoystickState()))

    def run(self):
        for _ in range(150):
            if self.controller._operation is None:
                return
            self.sample()
            if self.pending is not None:
                self.finish()
        pytest.fail("Direct alignment did not finish within the simulation bound")


@pytest.mark.parametrize("gain", [.25, -.25, 1, -1, 2.5, -2.5])
@pytest.mark.parametrize("error", [40, -40])
@pytest.mark.parametrize("slack", [0, 3, 7])
def test_direct_alignment_converges_without_calibration_despite_slack(gain, error, slack):
    # Whole-nudge backlash leaves these targets reachable on the discrete axis.
    axis = DirectAxis(error=error, gain=gain, slack=slack * 4, nudge=4)
    axis.run()
    assert "X aligned" in axis.controller.message
    assert abs(axis.error) <= CONTEXT.tolerance_px
    assert axis.steps[0] == -4
    assert {abs(steps) for steps in axis.steps} == {4}
    assert axis.controller.config == axis.original_config
    assert axis.controller.config.pixels_per_right_press is None


def test_default_nudge_probes_right_and_preserves_settings():
    axis = DirectAxis(config=MotionConfig(), nudge=1, slack=3, error=-20)
    axis.run()
    assert "X aligned" in axis.controller.message
    assert axis.steps[0] == -1
    assert set(map(abs, axis.steps)) == {1}
    assert axis.controller.config == MotionConfig()


def test_wrong_initial_direction_reverses_after_camera_reports_larger_error():
    axis = DirectAxis(error=-40)
    axis.sample()
    assert axis.pending.direction == Direction.RIGHT
    axis.finish()
    axis.sample()
    assert axis.pending.direction == Direction.LEFT


def test_same_side_error_increase_reverses_after_an_improving_nudge():
    axis = DirectAxis(error=12)
    axis.sample()
    axis.finish()
    axis.sample()
    assert axis.pending.direction == Direction.RIGHT
    axis.finish(actual=2)
    axis.sample()
    assert axis.pending.direction == Direction.LEFT


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("residual", [3, 14])
def test_settled_crossing_reverses_one_press_and_can_converge(direct, sign, residual):
    # 8 px before the first press, then either a larger error or less
    # than half a calibrated nudge on the opposite side of the reference.
    config = MotionConfig(pixels_per_right_press=None if direct else sign,
                          calibration_source_id=CONTEXT.source_id, calibration_shape=CONTEXT.shape)
    axis = DirectAxis(error=8 * sign, gain=sign, config=config, nudge=10, direct=direct)
    axis.sample()
    assert axis.pending.direction == Direction.RIGHT
    axis.finish(actual=-(8 + residual))
    assert axis.error == -residual * sign
    axis.sample(generation=axis.generation - 1)
    assert axis.pending is None
    axis.sample()
    assert axis.pending.direction == Direction.LEFT and "reversing" in axis.controller.message
    axis.finish(actual=residual)
    axis.run()
    assert axis.steps == [-10, 10] and "X aligned" in axis.controller.message
    assert axis.controller.config == config


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("travels", [
    (-20, 15, -5, 2),   # Both endpoints approach zero.
    (-28, 10, 18, -8),  # Revisits +8 from -10 after first crossing from +8 to -20.
])
def test_changing_overshoot_endpoints_can_reverse_in_both_directions(direct, travels):
    config = MotionConfig(pixels_per_right_press=None if direct else 1,
                          calibration_source_id=CONTEXT.source_id, calibration_shape=CONTEXT.shape)
    axis = DirectAxis(error=8, config=config, nudge=10, direct=direct)
    for travel in travels:
        axis.sample()
        assert axis.pending.direction == (Direction.RIGHT if travel < 0 else Direction.LEFT)
        axis.finish(actual=travel)
    axis.run()
    assert "X aligned" in axis.controller.message and len(axis.steps) == len(travels)


@pytest.mark.parametrize("direct", [False, True])
def test_repeated_unreachable_endpoints_stop_at_the_closer_endpoint(direct):
    config = MotionConfig(pixels_per_right_press=None if direct else 1,
                          calibration_source_id=CONTEXT.source_id, calibration_shape=CONTEXT.shape)
    axis = DirectAxis(error=8, config=config, nudge=10, direct=direct)
    axis.run()
    assert "one joystick nudge" in axis.controller.message
    assert axis.controller.armed and axis.error == -2
    assert axis.steps == [-10, 10, -10]


def test_saved_pixel_scale_does_not_control_direct_direction_or_correction():
    config = MotionConfig(pixels_per_right_press=-100, calibration_source_id=CONTEXT.source_id,
                          calibration_shape=CONTEXT.shape)
    axis = DirectAxis(config=config, gain=1, error=40)
    axis.run()
    assert "X aligned" in axis.controller.message
    assert axis.controller.config == config


def test_partial_driver_moves_and_position_wrap_do_not_break_feedback_search():
    axis = DirectAxis(error=-40)
    axis.position = 2**31 - 2
    axis.sample()
    assert axis.steps == [-4]
    axis.finish(actual=-3)
    axis.run()
    assert "X aligned" in axis.controller.message


def test_unresponsive_image_stops_without_disarming():
    axis = DirectAxis(gain=0)
    axis.run()
    assert "without progress" in axis.controller.message
    assert axis.controller._operation is None and axis.controller.armed


def test_unresolvable_tolerance_stops_instead_of_oscillating():
    axis = DirectAxis(error=4.5, nudge=1)
    axis.controller.context = replace(CONTEXT, tolerance_px=.1)
    axis.run()
    assert "one joystick nudge" in axis.controller.message
    assert axis.controller._operation is None


def test_completion_requires_three_distinct_valid_measurements_and_no_moves():
    axis = DirectAxis(error=.5)
    axis.sample()
    assert axis.controller._confirmations == 1
    axis.actions(axis.controller.tick(axis.now + .001, state()))
    assert axis.controller._confirmations == 1
    axis.sample()
    assert axis.controller._operation == "align_direct"
    axis.sample()
    assert "X aligned" in axis.controller.message
    assert axis.steps == []


@pytest.mark.parametrize("problem", ["old_generation", "inflight", "invalid", "uncertain", "nonfinite", "stale"])
def test_bad_feedback_never_starts_a_direct_move(problem):
    axis = DirectAxis(error=5)
    kwargs = {}
    if problem == "old_generation":
        kwargs["generation"] = 0
    elif problem == "inflight":
        kwargs["started"] = 0
    elif problem in {"invalid", "uncertain"}:
        kwargs.update(usable=False, validity=problem)
    elif problem == "stale":
        axis.now = 3
        value = feedback("stale", 5, 1.3, generation=axis.generation)
        assert not moves(axis.controller.tick(3, state(), feedback=value))
        assert axis.steps == []
        return
    axis.sample(math.nan if problem == "nonfinite" else 5, **kwargs)
    assert axis.steps == []
    axis.sample()
    assert axis.steps == [-4]


@pytest.mark.parametrize("interruption", ["stop", "manual", "context", "fault", "time", "corrections"])
def test_existing_interruptions_stop_direct_alignment(interruption):
    axis = DirectAxis()
    axis.sample()
    axis.finish()
    if interruption == "stop":
        axis.controller.cancel()
    elif interruption == "manual":
        axis.controller.set_keyboard(True, False)
    elif interruption == "context":
        axis.controller.set_context(replace(CONTEXT, reference_id="new"))
    elif interruption == "fault":
        axis.controller.tick(axis.now + .01, state(axis.position, fault="Overtemperature"))
    elif interruption == "time":
        axis.now += 61
    else:
        axis.controller._corrections = axis.controller.MAX_CORRECTIONS
    axis.sample()
    assert axis.controller._operation is None
    assert len(axis.steps) == 1


@pytest.mark.parametrize("missing", ["armed", "camera", "reference", "neutral"])
def test_direct_alignment_obeys_readiness_checks(missing):
    controller = ready()
    if missing == "armed":
        controller.armed = False
    elif missing == "camera":
        controller.set_context(replace(CONTEXT, paused=True))
    elif missing == "reference":
        controller.set_context(replace(CONTEXT, reference_id=None))
    else:
        controller.set_keyboard(True, False)
    controller.request_align_direct()
    assert not any(action.kind == "freshness" for action in controller.tick(1))
    assert controller._operation is None
