"""Calibration-free motor feedback with slack, unknown direction and scale."""
from dataclasses import replace
import math

import pytest

from measure_shift.motion import MotionConfig
from test_motion import CONTEXT, feedback, moves, ready, state


class DirectAxis:
    def __init__(self, *, error=40, gain=1, limit=32, slack=0, config=None):
        self.controller = ready(config or MotionConfig(jog_distance=limit))
        self.original_config = self.controller.config
        self.error = error
        self.gain = gain
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
        self.controller.request_align_direct()
        self.actions(self.controller.tick(self.now, state(self.position)))

    def actions(self, actions):
        for action in actions:
            if action.kind == "freshness":
                self.generation += 1
                self.cutoff = action.after
                self.controller.acknowledge_freshness(self.generation)
            elif action.kind == "move":
                assert self.pending is None
                self.pending = action
                self.steps.append(action.steps)
                assert action.velocity == self.controller.config.motor_speed
        return actions

    def sample(self, dx=None, **kwargs):
        self.now = max(self.now + .01, self.cutoff + .01)
        self.frame += 1
        kwargs.setdefault("generation", self.generation)
        value = feedback(self.frame, self.error if dx is None else dx, self.now, **kwargs)
        return self.actions(self.controller.tick(self.now, state(self.position), feedback=value))

    def finish(self, actual=None):
        assert self.pending is not None
        steps = self.pending.steps if actual is None else actual
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
        return self.actions(self.controller.tick(self.now, state(self.position)))

    def run(self):
        for _ in range(150):
            if self.controller._operation is None:
                return
            self.sample()
            if self.pending is not None:
                self.finish()
        pytest.fail("Direct alignment did not finish within the simulation bound")


@pytest.mark.parametrize("gain", [.2, -.2, 1, -1, 2.5, -2.5])
@pytest.mark.parametrize("error", [40, -40])
@pytest.mark.parametrize("slack", [0, 3, 7])
def test_direct_alignment_converges_without_calibration_despite_slack(gain, error, slack):
    axis = DirectAxis(error=error, gain=gain, slack=slack)
    axis.run()
    assert "X aligned" in axis.controller.message
    assert abs(axis.error) <= CONTEXT.tolerance_px
    assert axis.steps[0] == 4
    assert all(1 <= abs(steps) <= 32 for steps in axis.steps)
    assert axis.controller.config == axis.original_config
    assert axis.controller.config.pixels_per_step is None


def test_default_nudge_still_starts_with_four_steps_and_preserves_settings():
    axis = DirectAxis(config=MotionConfig(), slack=3, error=-20)
    axis.run()
    assert "X aligned" in axis.controller.message
    assert axis.steps[0] == 4
    assert max(map(abs, axis.steps)) == 4
    assert axis.controller.config == MotionConfig()


def test_growth_takes_up_slack_and_refinement_shrinks_after_crossing():
    axis = DirectAxis(error=-40, slack=6)
    axis.run()
    assert axis.steps[:4] == [4, 8, 16, 32]
    assert any(steps < 0 for steps in axis.steps)
    assert max(map(abs, axis.steps[4:])) < 32
    assert "X aligned" in axis.controller.message


def test_saved_pixel_scale_does_not_control_direct_direction_or_correction():
    config = MotionConfig(pixels_per_step=-100, calibration_source_id=CONTEXT.source_id,
                          calibration_shape=CONTEXT.shape, jog_distance=32)
    axis = DirectAxis(config=config, gain=1, error=40)
    axis.run()
    assert "X aligned" in axis.controller.message
    assert axis.controller.config == config


def test_partial_driver_moves_and_position_wrap_do_not_break_feedback_search():
    axis = DirectAxis(error=-40)
    axis.position = 2**31 - 2
    axis.sample()
    assert axis.steps == [4]
    axis.finish(actual=2)
    axis.run()
    assert "X aligned" in axis.controller.message


def test_zero_motor_travel_and_unresponsive_image_stop_without_disarming():
    axis = DirectAxis()
    axis.sample()
    axis.finish(actual=0)
    axis.sample()
    assert "motor position did not change" in axis.controller.message
    assert axis.controller._operation is None and axis.controller.armed

    axis = DirectAxis(gain=0)
    axis.run()
    assert "without progress" in axis.controller.message
    assert axis.controller._operation is None and axis.controller.armed


def test_unresolvable_tolerance_stops_instead_of_oscillating():
    axis = DirectAxis(error=4.5)
    axis.controller.context = replace(CONTEXT, tolerance_px=.1)
    axis.run()
    assert "one driver step" in axis.controller.message
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
    assert axis.steps == [4]


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
