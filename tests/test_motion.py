"""Single-axis behavior with deterministic time and a fake hardware transport."""
from dataclasses import replace
import math
import threading
import time

import pytest

from measure_shift.motion import MotionConfig, MotionContext, MotionController, MotionFeedback, MotionWorker
from measure_shift.motion_hardware import HardwareState


CONTEXT = MotionContext(source_id="ids:1", source_kind="IDS", shape=(32, 48),
                        reference_id="ref", config_id="match", connected=True)


def state(position=0, velocity=0, remaining=0, **kwargs):
    return HardwareState(position=position, velocity=velocity, remaining=remaining,
                         motor_current_ma=800, **kwargs)


def ready(config=None, context=CONTEXT):
    # Exercise custom wiring here; worker/UI tests cover the axis defaults.
    controller = MotionController(config or MotionConfig(motor_current_ma=800, left_channel=0, invert_direction=False))
    controller.set_context(context)
    controller.on_connected(state())
    controller.on_setup_applied()
    controller.arm()
    assert [action.kind for action in controller.tick(0)] == ["enable"]
    return controller


def moves(actions):
    return [action for action in actions if action.kind == "move"]


def feedback(frame, dx, received, *, generation=1, started=None, usable=True, **kwargs):
    return MotionFeedback(str(frame), "ref", "match", dx, usable, received,
                          received if started is None else started, generation, **kwargs)


def pixel_config(**kwargs):
    return MotionConfig(motor_current_ma=800, pixels_per_step=1,
                        calibration_source_id="ids:1", calibration_shape=(32, 48), **kwargs)


@pytest.mark.parametrize("field,value", [("repeat_hz", 0), ("repeat_hz", 11), ("jog_distance", math.nan),
                                          ("jog_distance", .6), ("jog_speed", .6),
                                          ("jog_speed", math.inf), ("left_channel", 4),
                                          ("reference_channel", -1), ("reference_channel", 4),
                                          ("reference_channel", True), ("reference_channel", "0"),
                                          ("reference_channel", 1.0),
                                          ("motor_current_ma", 100), ("step_resolution", 3),
                                          ("jog_unit", "mm"), ("pixels_per_step", 0)])
def test_invalid_configuration_is_rejected(field, value):
    with pytest.raises(ValueError):
        replace(MotionConfig(), **{field: value}).validated()


def test_configuration_roundtrip_and_pixel_unit_conversion():
    config = pixel_config(jog_unit="pixels", jog_distance=5, jog_speed=40, reference_channel=3)
    config = replace(config, pixels_per_step=-2)
    assert config.distance_steps() == 2
    assert config.speed_steps() == 20
    assert MotionConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="at least one"):
        replace(config, jog_distance=.1).validated()


@pytest.mark.parametrize("channel", range(4))
@pytest.mark.parametrize("active_high", [False, True])
def test_reference_capture_uses_raw_rising_edges_without_enabling_motion(channel, active_high):
    controller = MotionController(MotionConfig(reference_channel=channel, active_high=active_high))
    controller.on_connected(state())
    high = tuple(index == channel for index in range(4))
    low = (False, False, False, False)
    # Connecting while HIGH establishes the baseline without capturing.
    controller.tick(0, inputs=high)
    assert controller.snapshot().reference_capture is None
    controller.tick(.1, inputs=low)
    controller.tick(.2, inputs=high)
    first = controller.snapshot().reference_capture
    assert first.channel == channel and first.triggered_at == .2
    assert not controller.armed
    controller.tick(2.2, inputs=high)
    assert controller.snapshot().reference_capture == first
    controller.tick(2.3, inputs=low)
    assert controller.snapshot().reference_capture == first
    controller.tick(2.4, inputs=high)
    assert controller.snapshot().reference_capture.triggered_at == 2.4


def test_reference_capture_cooldown_discards_edges_without_replaying_held_high():
    controller = MotionController(MotionConfig(reference_channel=0))
    controller.on_connected(state())
    low = (False, False, False, False)
    high = (True, False, False, False)
    controller.tick(0, inputs=low)
    controller.tick(1, inputs=high)
    first = controller.snapshot().reference_capture
    controller.tick(1.1, inputs=low)
    controller.tick(1.999, inputs=high)
    assert controller.snapshot().reference_capture == first
    controller.tick(2, inputs=high)
    controller.tick(3, inputs=high)
    assert controller.snapshot().reference_capture == first
    controller.tick(3.1, inputs=low)
    controller.tick(3.2, inputs=high)
    assert controller.snapshot().reference_capture.triggered_at == 3.2
    controller.tick(4.1, inputs=low)
    controller.tick(4.2, inputs=high)
    assert controller.snapshot().reference_capture.triggered_at == 4.2


def test_reference_capture_disabled_and_other_channels_do_not_trigger():
    controller = MotionController()
    controller.on_connected(state())
    for now, inputs in enumerate(((False,) * 4, (True,) * 4) * 2):
        controller.tick(now, inputs=inputs)
        assert controller.snapshot().reference_capture is None
    assert MotionConfig.from_dict({}).reference_channel is None
    controller.configure(replace(controller.config, reference_channel=3), 1)
    controller.tick(4, inputs=(False,) * 4)
    controller.tick(5, inputs=(True, True, True, False))
    assert controller.snapshot().reference_capture is None
    controller.tick(6, inputs=(True,) * 4)
    assert controller.snapshot().reference_capture.channel == 3
    assert controller.snapshot().reference_capture.config_revision == 1


def test_reference_channel_changes_and_reconnects_reset_baseline_but_keep_cooldown():
    controller = MotionController(MotionConfig(reference_channel=0))
    controller.on_connected(state())
    controller.tick(0, inputs=(False,) * 4)
    controller.tick(1, inputs=(True, False, False, False))
    assert controller.snapshot().reference_capture is not None
    controller.configure(replace(controller.config, reference_channel=3), 1)
    controller.tick(1.1, inputs=(True,) * 4)
    assert controller.snapshot().reference_capture is None
    controller.tick(1.2, inputs=(False,) * 4)
    controller.tick(1.3, inputs=(True,) * 4)
    assert controller.snapshot().reference_capture is None
    controller.on_disconnected()
    controller.tick(1.4, inputs=(False,) * 4)
    controller.on_connected(state())
    controller.tick(1.5, inputs=(True,) * 4)
    assert controller.snapshot().reference_capture is None
    controller.tick(1.6, inputs=(False,) * 4)
    controller.tick(1.7, inputs=(True,) * 4)
    assert controller.snapshot().reference_capture is None
    controller.tick(1.9, inputs=(False,) * 4)
    controller.tick(2, inputs=(True,) * 4)
    assert controller.snapshot().reference_capture.triggered_at == 2
    controller.configure(replace(controller.config, reference_channel=None), 2)
    controller.tick(3, inputs=(False,) * 4)
    controller.tick(4, inputs=(True,) * 4)
    assert controller.snapshot().reference_capture is None


def test_legacy_mm_profile_converts_to_driver_steps():
    config = MotionConfig.from_dict({"jog_unit": "mm", "jog_distance": .05,
                                     "jog_speed": .5, "steps_per_mm": 800})
    assert config.jog_unit == "driversteps"
    assert config.distance_steps() == 40
    assert config.speed_steps() == 400
    assert "steps_per_mm" not in config.to_dict()


def test_current_defaults_to_800_in_new_and_previously_unset_profiles():
    assert MotionConfig().motor_current_ma == 800
    assert MotionConfig.from_dict({}).motor_current_ma == 800
    assert MotionConfig.from_dict({"motor_current_ma": None}).motor_current_ma == 800
    assert MotionConfig.from_dict({"motor_current_ma": 900}).motor_current_ma == 900


def test_enable_message_identifies_active_inputs_and_polarity():
    controller = MotionController(MotionConfig(active_high=False))
    controller.on_connected(state())
    controller.on_setup_applied()
    controller.arm()
    assert not controller.armed
    assert "channel 2: LOW" in controller.message
    assert "channel 1: LOW" in controller.message
    assert "check Active high" in controller.message
    controller.inputs4 = (True, True, True, True)
    controller.set_keyboard(True, False)
    controller.arm()
    assert "Left arrow key" in controller.message
    assert "check Active high" not in controller.message
    assert not controller.armed
    controller.set_keyboard(False, False)
    controller.arm()
    assert controller.armed and controller.phase == "ready"


def test_connection_and_setup_never_implicitly_arm():
    controller = MotionController()
    controller.on_connected(state())
    controller.set_keyboard(True, False)
    assert not moves(controller.tick(1))
    controller.arm()
    assert not controller.armed
    controller.on_setup_applied()
    controller.arm()
    assert not controller.armed
    controller.set_keyboard(False, False)
    controller.arm()
    assert controller.armed


def test_default_channels_drive_left_forward_and_right_backward():
    controller = MotionController()
    controller.on_connected(state())
    controller.on_setup_applied()
    controller.arm()
    controller.tick(0)
    # The unused channels must not be treated as joystick directions.
    assert not moves(controller.tick(.1, inputs=(True, False, False, True)))
    controller.tick(1, inputs=(False, False, True, False))
    assert [a.steps for a in moves(controller.tick(1.04, inputs=(False, False, True, False)))] == [1]
    controller.tick(1.1, state(1), (False, False, False, False))
    controller.tick(2, inputs=(False, True, False, False))
    assert [a.steps for a in moves(controller.tick(2.04, inputs=(False, True, False, False)))] == [-1]


def test_one_finite_nudge_per_press_release_finishes_it_and_repeat_is_ours():
    controller = ready()
    controller.set_keyboard(True, False)
    assert [a.steps for a in moves(controller.tick(1))] == [-1]
    controller.set_keyboard(False, False)
    assert controller.tick(1.1, state(velocity=-200, remaining=-1)) == ()
    assert controller.tick(1.2, state(-1)) == ()
    assert not moves(controller.tick(5, state(-1)))
    controller.set_keyboard(False, True)
    assert [a.steps for a in moves(controller.tick(6, state(-1)))] == [1]
    assert not moves(controller.tick(6.2, state(0)))
    assert [a.steps for a in moves(controller.tick(6.5, state(0)))] == [1]


def test_slow_moves_never_accumulate_a_repeat_queue():
    controller = ready()
    controller.set_keyboard(False, True)
    assert len(moves(controller.tick(1))) == 1
    for now in (1.5, 2, 2.5):
        assert not moves(controller.tick(now, state(velocity=1, remaining=1)))
    assert len(moves(controller.tick(3, state(1)))) == 1
    assert not moves(controller.tick(3.01, state(velocity=1, remaining=1)))


def test_joystick_debounce_and_keyboard_merge_avoid_double_nudges():
    controller = ready()
    assert not moves(controller.tick(1, inputs=(False, True, False, False)))
    assert not moves(controller.tick(1.02, inputs=(False, True, False, False)))
    assert len(moves(controller.tick(1.04, inputs=(False, True, False, False)))) == 1
    controller.set_keyboard(False, True)
    assert not moves(controller.tick(1.1, state(1), (False, True, False, False)))
    assert len(moves(controller.tick(1.54, state(1), (False, True, False, False)))) == 1


def test_joystick_reversal_cannot_repeat_the_previous_debounced_direction():
    controller = ready()
    controller.tick(1, inputs=(True, False, False, False))
    assert [a.steps for a in moves(controller.tick(1.04, inputs=(True, False, False, False)))] == [-1]
    assert not moves(controller.tick(1.54, state(-1), (False, True, False, False)))
    assert [a.steps for a in moves(controller.tick(1.58, state(-1), (False, True, False, False)))] == [1]


def test_released_contact_must_debounce_again_even_if_it_bounces_back():
    controller = ready()
    controller.tick(1, inputs=(True, False, False, False))
    assert len(moves(controller.tick(1.04, inputs=(True, False, False, False)))) == 1
    controller.tick(1.1, state(-1), (False, False, False, False))
    assert not moves(controller.tick(1.11, state(-1), (True, False, False, False)))
    assert not moves(controller.tick(1.12, state(-1), (False, False, False, False)))
    assert not moves(controller.tick(1.13, state(-1), (True, False, False, False)))
    assert not moves(controller.tick(1.15, state(-1), (True, False, False, False)))
    assert len(moves(controller.tick(1.17, state(-1), (True, False, False, False)))) == 1


def test_opposing_sources_interrupt_and_require_both_controls_released():
    controller = ready()
    controller.set_keyboard(True, False)
    controller.tick(1)
    actions = controller.tick(1.1, state(velocity=-100, remaining=-1), (False, True, False, False))
    assert [a.kind for a in actions] == ["stop"]
    controller.set_keyboard(False, False)
    assert not moves(controller.tick(1.2, state(-1), (False, True, False, False)))
    assert not moves(controller.tick(3, state(-1), (False, True, False, False)))
    controller.tick(3.1, state(-1), (False, False, False, False))
    controller.tick(3.2, state(-1), (False, False, False, False))
    controller.tick(3.3, state(-1), (False, True, False, False))
    assert [a.steps for a in moves(controller.tick(3.34, state(-1), (False, True, False, False)))] == [1]


def test_mapping_polarity_and_motor_direction_are_configurable():
    config = MotionConfig(motor_current_ma=800, left_channel=2, right_channel=3,
                          active_high=False, invert_direction=True)
    controller = MotionController(config)
    controller.on_connected(state())
    controller.inputs4 = (True, True, True, True)
    controller.tick(0, inputs=(True, True, True, True))
    controller.tick(.04, inputs=(True, True, True, True))
    controller.on_setup_applied()
    controller.arm()
    controller.tick(.1)
    controller.tick(1, inputs=(True, True, False, True))
    assert [a.steps for a in moves(controller.tick(1.04, inputs=(True, True, False, True)))] == [1]


def test_nudge_setting_edit_affects_next_move_without_interrupting_current_move():
    controller = ready()
    controller.set_keyboard(False, True)
    controller.tick(1)
    controller.configure(replace(controller.config, jog_distance=7, jog_speed=50, repeat_hz=4), 1)
    assert not controller._stopping
    assert not moves(controller.tick(1.2, state(velocity=200, remaining=1)))
    action = moves(controller.tick(1.5, state(1)))[0]
    assert (action.steps, action.velocity) == (7, 50)
    assert controller.snapshot().config_revision == 1


@pytest.mark.parametrize("disarm", [False, True])
def test_cancel_waits_for_motor_stop_and_requires_released_controls(disarm):
    controller = ready()
    controller.set_keyboard(False, True)
    controller.tick(1)
    if disarm:
        controller.cancel(disarm=True)
    else:
        controller.cancel()
    assert [a.kind for a in controller.tick(1.1, state(velocity=100, remaining=1))] == ["stop"]
    assert controller.armed == (not disarm)
    assert [a.kind for a in controller.tick(1.2, state(1))] == ["disable"]
    assert not moves(controller.tick(3, state(1)))
    controller.set_keyboard(False, False)
    assert not moves(controller.tick(3.1, state(1)))
    controller.set_keyboard(False, True)
    assert bool(moves(controller.tick(3.2, state(1)))) == (not disarm)


def test_cancel_does_not_arm_a_disabled_controller():
    controller = ready()
    controller.cancel(disarm=True)
    controller.tick(1)
    controller.cancel()
    controller.tick(2)
    controller.set_keyboard(False, True)
    assert not moves(controller.tick(3))
    assert not controller.armed


def test_stalled_motor_still_disarms_until_explicitly_enabled():
    controller = ready()
    controller.set_keyboard(False, True)
    controller.tick(1)
    moving = state(velocity=200, remaining=1)
    assert not controller.tick(5.9, moving)
    assert [action.kind for action in controller.tick(6.1, moving)] == ["stop"]
    assert not controller.armed
    assert "Motor move timed out" in controller.message
    assert [action.kind for action in controller.tick(6.2, state(1))] == ["disable"]
    controller.set_keyboard(False, False)
    controller.tick(6.3, state(1))
    controller.set_keyboard(False, True)
    assert not moves(controller.tick(6.4, state(1)))


def test_calibration_measures_actual_position_not_requested_steps():
    controller = ready()
    controller.request_calibration(10)
    actions = controller.tick(1, state(50))
    assert [a.kind for a in actions] == ["freshness"]
    cutoff = actions[0].after
    controller.acknowledge_freshness(1)
    action = moves(controller.tick(1.3, state(50), feedback=feedback("base", 4, cutoff + .01)))[0]
    assert action.steps == 10
    actions = controller.tick(1.4, state(58))
    assert [a.kind for a in actions] == ["freshness"]
    controller.acknowledge_freshness(2)
    controller.tick(1.7, state(58), feedback=feedback("after", 24, 1.61, generation=2))
    assert controller.config.pixels_per_step == 2.5
    assert controller.config.jog_unit == "pixels"
    assert controller.config.jog_distance == 2.5
    assert controller.config.jog_speed == 500
    assert "2.5 px/nudge" in controller.message
    assert controller.config.calibration_source_id == CONTEXT.source_id
    assert controller.config.calibration_shape == CONTEXT.shape
    assert controller.phase == "ready"


def test_calibration_rejects_small_pixel_motion_without_saving_scale():
    controller = ready()
    controller.request_calibration(10)
    controller.tick(1)
    controller.acknowledge_freshness(1)
    controller.tick(1.3, feedback=feedback("base", 0, 1.21))
    controller.tick(1.4, state(10))
    controller.acknowledge_freshness(2)
    actions = controller.tick(1.7, state(10), feedback=feedback("after", 1.9, 1.61, generation=2))
    assert [action.kind for action in actions] == ["stop"]
    assert controller.config.pixels_per_step is None
    assert "at least 2 pixels" in controller.message
    assert controller.armed and controller._operation is None

    # A rejected calibration can be retried without enabling the motor again.
    controller.request_calibration(20)
    actions = controller.tick(1.8, state(10))
    assert [action.kind for action in actions] == ["disable", "freshness"]
    controller.acknowledge_freshness(3)
    assert [action.steps for action in moves(controller.tick(
        2.1, state(10), feedback=feedback("retry", 1.9, 2.01, generation=3)))] == [20]


def test_recalibration_in_pixel_units_preserves_valid_motor_distance_and_speed():
    controller = ready(pixel_config(jog_unit="pixels", jog_distance=10, jog_speed=60_000))
    controller.request_calibration(10)
    controller.tick(1)
    controller.acknowledge_freshness(1)
    controller.tick(1.3, feedback=feedback("base", 0, 1.21))
    controller.tick(1.4, state(10))
    controller.acknowledge_freshness(2)
    controller.tick(1.7, state(10), feedback=feedback("after", 2, 1.61, generation=2))
    assert controller.config.pixels_per_step == .2
    assert controller.config.jog_unit == "pixels"
    assert controller.config.jog_distance == 2
    assert controller.config.jog_speed == 12_000
    assert controller.config.distance_steps() == 10
    assert controller.config.speed_steps() == 60_000
    assert controller.config.validated() == controller.config


def test_alignment_converges_with_bounded_moves_and_three_new_confirmations():
    controller = ready(pixel_config(jog_distance=2))
    controller.request_align()
    controller.tick(1)
    controller.acknowledge_freshness(1)
    assert [a.steps for a in moves(controller.tick(1.3, feedback=feedback("a", 5, 1.21)))] == [-2]
    assert [a.kind for a in controller.tick(1.4, state(-2))] == ["freshness"]
    controller.acknowledge_freshness(2)
    assert [a.steps for a in moves(controller.tick(1.7, state(-2), feedback=feedback("b", 3, 1.61, generation=2)))] == [-2]
    controller.tick(1.8, state(-4))
    controller.acknowledge_freshness(3)
    assert not moves(controller.tick(2.1, state(-4), feedback=feedback("c", 1, 2.01, generation=3)))
    assert controller._confirmations == 1
    controller.tick(2.11, state(-4))
    assert controller._confirmations == 1
    controller.tick(2.2, state(-4), feedback=feedback("d", .9, 2.2, generation=3))
    controller.tick(2.3, state(-4), feedback=feedback("e", .8, 2.3, generation=3))
    assert controller._operation is None
    assert "X aligned" in controller.message
    assert controller.snapshot().corrections == 2


def test_negative_calibration_reverses_automatic_motor_direction():
    controller = ready(replace(pixel_config(jog_distance=3), pixels_per_step=-2))
    controller.request_align()
    controller.tick(1)
    controller.acknowledge_freshness(1)
    assert [a.steps for a in moves(controller.tick(1.3, feedback=feedback("a", 8, 1.21)))] == [3]


def test_old_inflight_and_wrong_generation_frames_cannot_move_motor():
    controller = ready(pixel_config())
    controller.request_align()
    controller.tick(1)
    controller.acknowledge_freshness(5)
    for value in (feedback("old", 10, 1.3, started=.9, generation=5),
                  feedback("token", 10, 1.31, generation=4)):
        assert not moves(controller.tick(1.4, feedback=value))
    assert len(moves(controller.tick(1.5, feedback=feedback("good", 10, 1.4, generation=5)))) == 1


@pytest.mark.parametrize("change", [dict(paused=True), dict(connected=False), dict(reference_id="new"),
                                     dict(config_id="new"), dict(source_id="ids:2"), dict(shape=(64, 48)),
                                     dict(tolerance_px=3)])
def test_context_change_cancels_feedback_motion(change):
    controller = ready(pixel_config())
    controller.request_align()
    controller.tick(1)
    controller.set_context(replace(CONTEXT, **change))
    assert controller._operation is None
    assert any(a.kind == "stop" for a in controller.tick(1.1))


def test_camera_change_invalidates_pixel_scale_and_preserves_realized_nudge():
    controller = ready(replace(pixel_config(), jog_unit="pixels", jog_distance=6, jog_speed=400,
                               pixels_per_step=2))
    controller.set_context(replace(CONTEXT, source_id="ids:2"))
    assert controller.config.pixels_per_step is None
    assert controller.config.jog_unit == "driversteps"
    assert controller.config.jog_distance == 3
    assert controller.config.jog_speed == 200


def test_microstep_change_invalidates_pixel_calibration():
    controller = ready(replace(pixel_config(), jog_unit="pixels", jog_distance=10, jog_speed=200))
    controller.configure(replace(controller.config, step_resolution=16), 7)
    assert controller.config.pixels_per_step is None
    assert controller.config.jog_unit == "driversteps"
    assert controller.config.jog_distance == 10
    assert not controller.armed
    assert not controller.setup_applied


def test_external_microstep_change_stops_and_invalidates_calibrations():
    controller = ready(pixel_config())
    actions = controller.tick(1, state(step_resolution=16))
    assert "microstep resolution changed" in controller.message
    assert any(action.kind == "stop" for action in actions)
    assert not controller.armed
    assert not controller.setup_applied
    assert controller.config.pixels_per_step is None


@pytest.mark.parametrize("kind", ["Demo", "Images", "Video"])
def test_replay_and_demo_sources_cannot_drive_auto_alignment(kind):
    controller = ready(pixel_config(), replace(CONTEXT, source_kind=kind))
    controller.request_align()
    assert not moves(controller.tick(1))
    assert controller._operation is None
    assert "IDS or Basler" in controller.message


@pytest.mark.parametrize("dx,usable,received,message", [(2, False, 1.21, "invalid"), (2, True, 1.21, "stale")])
def test_invalid_or_stale_measurements_retry_before_stopping_alignment(dx, usable, received, message):
    controller = ready(pixel_config())
    controller.request_align()
    controller.tick(1)
    controller.acknowledge_freshness(1)
    now = 2.4 if message == "stale" else 1.3
    actions = controller.tick(now, feedback=feedback("bad", dx, received, usable=usable))
    assert not moves(actions) and not any(action.kind == "stop" for action in actions)
    assert message in controller.message
    assert controller.armed and controller._operation == "align"
    controller.tick(now + 3)
    assert controller.armed and controller._operation is None

    # Stopping the failed attempt does not resume it when feedback returns.
    assert not moves(controller.tick(now + 3.1))
    assert controller.phase == "ready"
    controller.request_align()
    actions = controller.tick(now + 3.2)
    assert [action.kind for action in actions] == ["freshness"]
    controller.acknowledge_freshness(2)
    assert [action.steps for action in moves(controller.tick(
        now + 3.5, feedback=feedback("retry", 2, actions[0].after + .01, generation=2)))] == [-1]


@pytest.mark.parametrize("dx,dy", [(11, 0), (-11, 0), (3, 11), (3, -11), (-3, 11)])
def test_uncertain_large_offset_allows_bounded_x_correction(dx, dy):
    controller = ready(pixel_config(jog_distance=2))
    controller.request_align()
    controller.tick(1)
    controller.acknowledge_freshness(1)
    actions = controller.tick(1.3, feedback=feedback(
        "coarse", dx, 1.21, dy=dy, usable=False, validity="uncertain"))
    assert [action.steps for action in moves(actions)] == [-2 if dx > 0 else 2]
    assert controller.armed and controller._operation == "align"
    assert "uncertain match at large offset" in controller.message


@pytest.mark.parametrize("dx,dy", [(0, 0), (1, -1), (10, 0), (0, -10), (8, 8)])
def test_uncertain_low_offset_blocks_alignment(dx, dy):
    controller = ready(pixel_config())
    controller.request_align()
    controller.tick(1)
    controller.acknowledge_freshness(1)
    actions = controller.tick(1.3, feedback=feedback(
        "near", dx, 1.21, dy=dy, usable=False, validity="uncertain"))
    assert not moves(actions)
    assert not any(action.kind == "stop" for action in actions)
    assert controller.armed and controller._operation == "align"
    actions = controller.tick(4.3)
    assert any(action.kind == "stop" for action in actions)
    assert controller.armed and controller._operation is None
    assert "uncertain at low offset" in controller.message


@pytest.mark.parametrize("dx,dy", [(100, 0), (math.nan, 100), (100, math.inf)])
def test_invalid_or_nonfinite_large_offsets_still_block_alignment(dx, dy):
    controller = ready(pixel_config())
    controller.request_align()
    controller.tick(1)
    controller.acknowledge_freshness(1)
    validity = "invalid" if math.isfinite(dx) and math.isfinite(dy) else "uncertain"
    assert not moves(controller.tick(1.3, feedback=feedback(
        "bad", dx, 1.21, dy=dy, usable=False, validity=validity)))
    assert controller.armed
    controller.tick(4.3)
    assert controller.armed and controller._operation is None


def test_uncertain_offset_must_also_exceed_tolerance():
    controller = ready(pixel_config(), replace(CONTEXT, tolerance_px=20))
    controller.request_align()
    controller.tick(1)
    controller.acknowledge_freshness(1)
    assert not moves(controller.tick(1.3, feedback=feedback(
        "near", 15, 1.21, usable=False, validity="uncertain")))
    assert controller.armed
    controller.tick(4.3)
    assert controller.armed and controller._operation is None


def test_uncertain_coarse_alignment_waits_for_valid_match_when_offsets_become_small():
    controller = ready(pixel_config(jog_distance=2))
    controller.request_align()
    controller.tick(1)
    controller.acknowledge_freshness(1)
    assert moves(controller.tick(1.3, feedback=feedback(
        "coarse", 11, 1.21, usable=False, validity="uncertain")))
    controller.tick(1.4, state(-2))
    controller.acknowledge_freshness(2)
    actions = controller.tick(1.7, state(-2), feedback=feedback(
        "near", 9, 1.61, generation=2, usable=False, validity="uncertain"))
    assert not moves(actions) and controller.armed
    assert "uncertain at low offset" in controller.message
    assert moves(controller.tick(1.8, state(-2), feedback=feedback("valid", 9, 1.79, generation=2)))


@pytest.mark.parametrize("dx,dy,allowed", [(11, 0, True), (3, 11, True), (3, 0, False)])
def test_uncertain_policy_also_applies_during_motor_movement(dx, dy, allowed):
    controller = ready(pixel_config(jog_distance=2))
    controller.request_align()
    controller.tick(1)
    controller.acknowledge_freshness(1)
    assert moves(controller.tick(1.3, feedback=feedback("base", 12, 1.21)))
    actions = controller.tick(1.4, state(-1, velocity=200, remaining=1), feedback=feedback(
        "moving", dx, 1.35, dy=dy, usable=False, validity="uncertain"))
    assert controller.armed
    assert not any(action.kind == "stop" for action in actions)
    actions = controller.tick(4.4, state(-1, velocity=200, remaining=1), feedback=feedback(
        "later", dx, 4.35, dy=dy, usable=False, validity="uncertain"))
    assert controller.armed
    assert (controller._operation == "align") == allowed
    assert any(action.kind == "stop" for action in actions) == (not allowed)


@pytest.mark.parametrize("operation", ["align", "calibration"])
@pytest.mark.parametrize("problem", ["invalid", "uncertain", "nonfinite", "stale"])
def test_transient_feedback_failure_during_movement_recovers(operation, problem):
    controller = ready(pixel_config(jog_distance=2))
    if operation == "align":
        controller.request_align()
    else:
        controller.request_calibration(2)
    controller.tick(1)
    controller.acknowledge_freshness(1)
    assert moves(controller.tick(1.3, feedback=feedback("base", 12, 1.21)))
    moving = state(velocity=200, remaining=1)
    bad = feedback("bad", math.nan if problem == "nonfinite" else 2,
                   1.21 if problem == "stale" else 2.39,
                   usable=problem == "stale", validity=problem)
    actions = controller.tick(2.4, moving, feedback=bad)
    assert not moves(actions) and not any(action.kind == "stop" for action in actions)
    assert controller.armed and "retrying camera feedback" in controller.message
    actions = controller.tick(2.5, moving, feedback=feedback("recovered", 10, 2.49))
    assert not actions
    assert controller.armed and "recovered" in controller.message

    position = -2 if operation == "align" else 2
    assert [action.kind for action in controller.tick(2.6, state(position))] == ["freshness"]
    controller.acknowledge_freshness(2)
    actions = controller.tick(2.9, state(position), feedback=feedback(
        "after", 10 if operation == "align" else 16, 2.81, generation=2))
    if operation == "align":
        assert [action.steps for action in moves(actions)] == [-2]
    else:
        assert controller.config.pixels_per_step == 2
        assert controller._operation is None
    assert controller.armed


def test_repeated_bad_frames_and_motor_stop_do_not_extend_feedback_retry_deadline():
    controller = ready(pixel_config(jog_distance=2))
    controller.request_align()
    controller.tick(1)
    controller.acknowledge_freshness(1)
    controller.tick(1.3, feedback=feedback("base", 12, 1.21))
    moving = state(-1, velocity=200, remaining=1)
    controller.tick(1.4, moving, feedback=feedback("bad", 2, 1.39, usable=False))
    for now in (1.41, 1.42, 1.43, 1.44, 1.45):
        assert not controller.tick(now, moving)
        assert controller.armed
    controller.tick(1.6, state(-2))
    controller.acknowledge_freshness(2)
    for now in (1.9, 2.5, 3.5, 4.39):
        assert not controller.tick(now, state(-2), feedback=feedback(
            str(now), 2, now - .01, generation=2, usable=False))
        assert controller.armed
    actions = controller.tick(4.4, state(-2), feedback=feedback(
        "last", 2, 4.39, generation=2, usable=False))
    assert any(action.kind == "stop" for action in actions)
    assert controller.armed and controller._operation is None
    assert "3 seconds" in controller.message


def test_feedback_retry_deadline_still_applies_while_waiting_for_postmove_freshness():
    controller = ready(pixel_config(jog_distance=2))
    controller.request_align()
    controller.tick(1)
    controller.acknowledge_freshness(1)
    controller.tick(1.3, feedback=feedback("base", 12, 1.21))
    controller.tick(1.4, state(-1, velocity=200, remaining=1),
                    feedback=feedback("bad", 2, 1.39, usable=False))
    controller.tick(4.2, state(-2))
    controller.acknowledge_freshness(2)
    assert controller.armed
    actions = controller.tick(4.41, state(-2))
    assert any(action.kind == "stop" for action in actions)
    assert controller.armed and controller._operation is None


def test_good_feedback_resets_the_retry_window_during_movement():
    controller = ready(pixel_config(jog_distance=2))
    controller.request_align()
    controller.tick(1)
    controller.acknowledge_freshness(1)
    controller.tick(1.3, feedback=feedback("base", 12, 1.21))
    moving = state(-1, velocity=200, remaining=1)
    controller.tick(1.4, moving, feedback=feedback("bad1", 2, 1.39, usable=False))
    controller.tick(2, moving, feedback=feedback("good", 10, 1.99))
    controller.tick(3, moving, feedback=feedback("bad2", 2, 2.99, usable=False))
    assert not controller.tick(4.5, moving)
    assert controller.armed
    actions = controller.tick(6, moving)
    assert any(action.kind == "stop" for action in actions)
    assert controller.armed and controller._operation is None


def test_retry_requires_three_valid_alignment_confirmations_after_recovery():
    controller = ready(pixel_config())
    controller.request_align()
    controller.tick(1)
    controller.acknowledge_freshness(1)
    controller.tick(1.3, feedback=feedback("valid", .2, 1.21))
    controller.tick(1.4, feedback=feedback("bad", .2, 1.39, usable=False))
    assert controller.armed and controller._confirmations == 0
    for index, now in enumerate((1.5, 1.6)):
        assert not moves(controller.tick(now, feedback=feedback(f"good{index}", .2, now - .01)))
        assert controller._operation == "align"
    controller.tick(1.7, feedback=feedback("final", .2, 1.69))
    assert controller._operation is None and "X aligned" in controller.message


def test_uncertain_large_offsets_still_require_fresh_feedback():
    controller = ready(pixel_config())
    controller.request_align()
    controller.tick(1)
    controller.acknowledge_freshness(2)
    for value in (feedback("old", 100, 1.3, started=.9, generation=2, usable=False, validity="uncertain"),
                  feedback("token", 100, 1.31, generation=1, usable=False, validity="uncertain")):
        assert not moves(controller.tick(1.4, feedback=value))
    assert controller.armed
    assert not moves(controller.tick(2.4, feedback=feedback(
        "stale", 100, 1.21, generation=2, usable=False, validity="uncertain")))
    assert controller.armed and "stale" in controller.message
    controller.tick(5.4)
    assert controller.armed and controller._operation is None
    assert "stale" in controller.message


def test_uncertain_large_y_cannot_confirm_x_alignment():
    controller = ready(pixel_config())
    controller.request_align()
    controller.tick(1)
    controller.acknowledge_freshness(1)
    controller.tick(1.3, feedback=feedback("valid", .2, 1.21))
    assert controller._confirmations == 1
    for index in range(3):
        now = 1.4 + index * .1
        assert not moves(controller.tick(now, feedback=feedback(
            index, .2, now - .01, dy=11, usable=False, validity="uncertain")))
        assert controller._confirmations == 0
        assert controller._operation == "align"
    assert "waiting for a valid match" in controller.message
    for index in range(3):
        now = 1.8 + index * .1
        controller.tick(now, feedback=feedback(f"valid{index}", .2, now - .01, dy=11))
    assert controller._operation is None and "X aligned" in controller.message


def test_uncertain_large_offset_does_not_allow_pixel_calibration():
    controller = ready()
    controller.request_calibration(100)
    controller.tick(1)
    controller.acknowledge_freshness(1)
    assert not moves(controller.tick(1.3, feedback=feedback(
        "coarse", 100, 1.21, usable=False, validity="uncertain")))
    assert controller.armed and controller.config.pixels_per_step is None
    controller.tick(4.3)
    assert controller.armed and controller._operation is None
    assert controller.config.pixels_per_step is None


def test_motion_feedback_preserves_uncertainty_and_both_offsets():
    import numpy as np
    from measure_shift.models import FramePacket, InspectionPair, MatchConfig, MatchResult, ReferenceSnapshot

    frame = FramePacket.from_array(np.zeros((8, 8), dtype=np.uint8))
    reference, config = ReferenceSnapshot(frame), MatchConfig()
    result = MatchResult(frame.frame_id, reference.reference_id, config.config_id,
                         dx=3, dy=-15, validity="uncertain")
    value = MotionFeedback.from_pair(InspectionPair(reference, frame, config, result))
    assert (value.dx, value.dy, value.validity, value.usable) == (3, -15, "uncertain", False)


@pytest.mark.parametrize("errors,message", [((5, 6), "increased"), ((5,) * 11, "10 moves without progress")])
def test_increasing_error_and_ten_no_progress_results_stop_alignment_without_disarming(errors, message):
    controller = ready(pixel_config())
    controller.request_align()
    controller.tick(1)
    position = 0
    for index, error in enumerate(errors):
        controller.acknowledge_freshness(index + 1)
        now = 1.3 + index * .5
        actions = controller.tick(now, state(position), feedback=feedback(index, error, now - .01, generation=index + 1))
        if index != len(errors) - 1:
            assert moves(actions)
            position -= 1
            controller.tick(now + .1, state(position))
    assert message in controller.message
    assert controller.armed and controller._operation is None


def test_manual_input_interrupts_auto_and_motor_fault_disarms():
    controller = ready(pixel_config())
    controller.request_align()
    controller.tick(1)
    controller.set_keyboard(True, False)
    assert any(a.kind == "stop" for a in controller.tick(1.1))
    assert "Manual input" in controller.message
    controller.set_keyboard(False, False)
    controller.tick(1.2)
    controller.tick(1.3, state(fault="Overtemperature"))
    assert not controller.armed
    assert "Overtemperature" in controller.message


class FakeHardware:
    def __init__(self):
        self.calls = []
        self.state = state()
        self.inputs = (False, False, False, False)
        self.fail = False

    def _record(self, name, *args):
        self.calls.append((name, args, threading.get_ident()))
        if self.fail:
            raise RuntimeError("fake connection lost")

    def connect(self, *args): self._record("connect", *args)
    def read_inputs(self):
        self._record("inputs")
        return self.inputs
    def read_state(self):
        self._record("state")
        return self.state
    def configure(self, current, resolution, interpolation):
        self._record("configure", current, resolution, interpolation)
        self.state = replace(self.state, motor_current_ma=current, step_resolution=resolution, interpolation=interpolation)
    def enable(self):
        self._record("enable")
        self.state = replace(self.state, enabled=True)
    def disable(self):
        self._record("disable")
        self.state = replace(self.state, enabled=False)
    def move_steps(self, steps, velocity):
        self._record("move", steps, velocity)
        self.state = replace(self.state, position=self.state.position + steps)
    def stop(self): self._record("stop")
    def shutdown(self, timeout_s=2):
        self._record("shutdown")
        self.state = replace(self.state, enabled=False, velocity=0, remaining=0)


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        threading.Event().wait(.005)
    pytest.fail("Motion worker did not reach the expected state")


@pytest.mark.parametrize("operation", ["align", "calibration"])
def test_worker_retries_camera_failure_and_restores_motor_power_without_enable_click(operation):
    hardware = FakeHardware()
    barriers = []

    def request(after):
        barriers.append(after)
        return 0 if len(barriers) == 1 else len(barriers)

    worker = MotionWorker(pixel_config(), hardware=hardware, freshness_callback=request)
    worker.set_context(CONTEXT)
    retry = worker.align if operation == "align" else lambda: worker.calibrate(2)
    worker.start()
    try:
        worker.connect_hardware()
        wait_for(lambda: worker.output.read()[0].phase == "ready")
        retry()
        wait_for(lambda: "freshness barrier is unavailable" in worker.output.read()[0].message
                 and worker.output.read()[0].phase == "ready")
        assert worker.output.read()[0].armed
        assert not hardware.state.enabled
        assert not any(call[0] == "move" for call in hardware.calls)

        retry()
        wait_for(lambda: len(barriers) == 2 and time.monotonic() > barriers[-1])
        worker.submit_feedback(feedback("retry", 2, time.monotonic(), generation=2))
        wait_for(lambda: any(call[0] == "move" for call in hardware.calls))
        assert worker.output.read()[0].armed and hardware.state.enabled
        move_index = next(i for i, call in enumerate(hardware.calls) if call[0] == "move")
        assert hardware.calls[move_index - 1][0] == "enable"
        assert hardware.calls[move_index][1][0] == (-1 if operation == "align" else 2)
    finally:
        worker.stop_worker()
        worker.join(timeout=2)


def test_worker_keeps_all_adapter_io_off_caller_thread_and_never_moves_on_connect():
    hardware = FakeHardware()
    worker = MotionWorker(MotionConfig(motor_current_ma=800), hardware=hardware)
    worker.start()
    try:
        worker.connect_hardware()
        wait_for(lambda: worker.output.read()[0].phase == "ready")
        assert hardware.state.enabled
        assert not any(call[0] == "move" for call in hardware.calls)
        worker.apply_setup()
        wait_for(lambda: worker.output.read()[0].setup_applied and not worker.output.read()[0].armed)
        worker.arm()
        wait_for(lambda: worker.output.read()[0].armed)
        worker.set_keyboard(False, True)
        wait_for(lambda: any(call[0] == "move" for call in hardware.calls))
        worker.set_keyboard(False, False)
        assert all(call[2] != threading.get_ident() for call in hardware.calls)
        revision = worker.configure(replace(worker.controller.config, jog_distance=2))
        wait_for(lambda: worker.output.read()[0].config_revision == revision)
        worker.stop()
        wait_for(lambda: any(call[0] == "stop" for call in hardware.calls))
        assert worker.output.read()[0].armed
    finally:
        worker.stop_worker()
        worker.join(timeout=2)
    assert not worker.is_alive()
    assert hardware.calls[-1][0] == "shutdown"


@pytest.mark.parametrize("final_keyboard,expected_steps", [((False, False), []), ((False, True), [-1])])
def test_worker_drops_input_history_during_blocked_hardware_read(final_keyboard, expected_steps):
    hardware = BlockingHardware()
    worker = MotionWorker(MotionConfig(motor_current_ma=800), hardware=hardware)
    worker.start()
    try:
        worker.connect_hardware()
        wait_for(lambda: worker.output.read()[0].connected)
        worker.apply_setup()
        wait_for(lambda: worker.output.read()[0].setup_applied and not worker.output.read()[0].armed)
        worker.arm()
        wait_for(lambda: worker.output.read()[0].armed)
        hardware.block = True
        assert hardware.entered.wait(1)
        for _ in range(100):
            worker.set_keyboard(True, False)
            worker.set_keyboard(False, False)
        worker.set_keyboard(*final_keyboard)
        hardware.release.set()
        if expected_steps:
            wait_for(lambda: any(call[0] == "move" for call in hardware.calls))
            worker.set_keyboard(False, False)
        threading.Event().wait(.15)
        assert [call[1][0] for call in hardware.calls if call[0] == "move"] == expected_steps
    finally:
        hardware.release.set()
        worker.stop_worker()
        worker.join(timeout=2)


def test_released_keyboard_tap_is_not_retained_for_a_later_tick():
    controller = ready()
    controller.set_keyboard(True, False)
    controller.set_keyboard(False, False)
    assert not moves(controller.tick(1))
    assert not moves(controller.tick(10))


def test_taps_during_a_move_are_not_replayed_after_completion():
    controller = ready()
    controller.set_keyboard(True, False)
    assert len(moves(controller.tick(1))) == 1
    controller.set_keyboard(False, False)
    for _ in range(10):
        controller.set_keyboard(False, True)
        controller.set_keyboard(False, False)
    assert not moves(controller.tick(1.1, state(velocity=-200, remaining=-1)))
    assert not moves(controller.tick(2, state(-1)))
    assert not moves(controller.tick(10, state(-1)))
    controller.set_keyboard(False, True)
    assert [a.steps for a in moves(controller.tick(10.1, state(-1)))] == [1]


def test_stop_requires_neutral_before_a_new_press():
    controller = ready()
    controller.set_keyboard(True, False)
    controller.set_keyboard(False, False)
    controller.cancel("Stopped", disarm=False)
    assert not moves(controller.tick(1))
    assert controller.armed
    controller.set_keyboard(True, False)
    assert [a.steps for a in moves(controller.tick(1.1))] == [-1]


@pytest.mark.parametrize("keys", [
    [(True, False), (False, False)],
    [(True, False), (True, True), (False, False)],
])
def test_latest_keyboard_state_keeps_transient_conflict_and_manual_override(keys):
    worker = MotionWorker()
    worker.controller = ready(pixel_config())
    worker.controller.request_align()
    worker.controller.tick(1)
    for keyboard in keys:
        worker.set_keyboard(*keyboard)
    worker._sync_keyboard()
    actions = worker.controller.tick(1.1)
    assert [action.kind for action in actions] == ["stop", "disable"]
    assert worker.controller._operation is None
    assert worker.controller.keyboard == (False, False)


def test_settling_time_cannot_be_less_than_two_hundred_milliseconds():
    controller = ready(pixel_config(), replace(CONTEXT, settling_s=0))
    controller.request_align()
    action = controller.tick(1)[0]
    assert action.kind == "freshness"
    assert action.after == pytest.approx(1.2)


def test_worker_connection_failure_disarms_without_replaying_motion():
    hardware = FakeHardware()
    worker = MotionWorker(MotionConfig(motor_current_ma=800), hardware=hardware)
    worker.start()
    try:
        worker.connect_hardware()
        wait_for(lambda: worker.output.read()[0].connected)
        hardware.fail = True
        wait_for(lambda: not worker.output.read()[0].connected)
        assert not worker.output.read()[0].armed
        assert "fake connection lost" in worker.output.read()[0].message
        assert not any(call[0] == "move" for call in hardware.calls)
    finally:
        hardware.fail = False
        worker.stop_worker()
        worker.join(timeout=2)


def test_worker_arming_reads_fresh_inputs_and_setup_disables_power():
    hardware = FakeHardware()
    worker = MotionWorker(MotionConfig(motor_current_ma=800), hardware=hardware)
    worker.start()
    try:
        worker.connect_hardware()
        wait_for(lambda: worker.output.read()[0].phase == "ready")
        worker.apply_setup()
        wait_for(lambda: worker.output.read()[0].setup_applied and not worker.output.read()[0].armed)
        enables = sum(call[0] == "enable" for call in hardware.calls)
        hardware.inputs = (False, False, True, False)
        worker.arm()
        wait_for(lambda: "Release controls" in worker.output.read()[0].message)
        assert "Left joystick input (channel 2: HIGH)" in worker.output.read()[0].message
        assert not worker.output.read()[0].armed
        assert sum(call[0] == "enable" for call in hardware.calls) == enables
        assert not any(call[0] == "move" for call in hardware.calls)
        hardware.inputs = (False, False, False, False)
        worker.arm()
        wait_for(lambda: worker.output.read()[0].armed)
        worker.apply_setup()
        wait_for(lambda: not worker.output.read()[0].armed)
        assert not hardware.state.enabled
        config_index = max(i for i, call in enumerate(hardware.calls) if call[0] == "configure")
        assert hardware.calls[config_index - 1][0] == "disable"
    finally:
        worker.stop_worker()
        worker.join(timeout=2)


def test_connect_applies_default_setup_and_enable_reapplies_edited_settings_without_moving():
    hardware = FakeHardware()
    worker = MotionWorker(hardware=hardware)
    worker.start()
    try:
        worker.connect_hardware()
        wait_for(lambda: worker.output.read()[0].phase == "ready")
        assert hardware.state.motor_current_ma == 800 and hardware.state.enabled
        assert ("configure", (800, 8, True)) in [(name, args) for name, args, thread in hardware.calls]
        revision = worker.configure(replace(worker.controller.config, motor_current_ma=900))
        wait_for(lambda: worker.output.read()[0].config_revision == revision and not worker.output.read()[0].armed)
        worker.arm()
        wait_for(lambda: worker.output.read()[0].armed)
        assert hardware.state.motor_current_ma == 900 and hardware.state.enabled
        assert not any(call[0] == "move" for call in hardware.calls)
    finally:
        worker.stop_worker()
        worker.join(timeout=2)


@pytest.mark.parametrize("inputs,keyboard", [
    ((False, False, True, False), (False, False)),
    ((False, True, False, False), (False, False)),
    ((False, True, True, False), (False, False)),
    ((False, False, False, False), (True, False)),
])
def test_connect_with_held_controls_waits_for_release_then_a_new_press(inputs, keyboard):
    hardware = FakeHardware()
    hardware.inputs = inputs
    worker = MotionWorker(hardware=hardware)
    worker.set_keyboard(*keyboard)
    worker.start()
    try:
        worker.connect_hardware()
        wait_for(lambda: worker.output.read()[0].phase == "waiting")
        assert worker.output.read()[0].armed and hardware.state.enabled
        threading.Event().wait(.08)  # Longer than input debounce.
        assert not any(call[0] == "move" for call in hardware.calls)
        hardware.inputs = (False, False, False, False)
        worker.set_keyboard(False, False)
        wait_for(lambda: worker.output.read()[0].phase == "ready")
        assert not any(call[0] == "move" for call in hardware.calls)
        hardware.inputs = (False, False, True, False)
        wait_for(lambda: any(call[0] == "move" for call in hardware.calls))
        assert [call[1][0] for call in hardware.calls if call[0] == "move"] == [1]
    finally:
        worker.stop_worker()
        worker.join(timeout=2)


def test_key_tap_during_connection_is_not_replayed_after_connect():
    entered, release = threading.Event(), threading.Event()

    class OpeningHardware(FakeHardware):
        def connect(self, *args):
            super().connect(*args)
            entered.set()
            assert release.wait(1)

    hardware = OpeningHardware()
    worker = MotionWorker(hardware=hardware)
    worker.start()
    try:
        worker.connect_hardware()
        assert entered.wait(1)
        worker.set_keyboard(True, False)
        worker.set_keyboard(False, False)
        release.set()
        wait_for(lambda: worker.output.read()[0].phase == "ready")
        assert not any(call[0] == "move" for call in hardware.calls)
        worker.set_keyboard(True, False)
        wait_for(lambda: any(call[0] == "move" for call in hardware.calls))
        assert [call[1][0] for call in hardware.calls if call[0] == "move"] == [1]
    finally:
        release.set()
        worker.stop_worker()
        worker.join(timeout=2)


@pytest.mark.parametrize("failure", ["configure", "fault"])
def test_connect_does_not_enable_motor_when_setup_fails_or_driver_has_a_fault(failure):
    class UnhealthyHardware(FakeHardware):
        def configure(self, *args):
            if failure == "configure":
                raise RuntimeError("Cannot configure motor")
            super().configure(*args)

    hardware = UnhealthyHardware()
    if failure == "fault":
        hardware.state = replace(hardware.state, fault="Motor driver short to ground")
    worker = MotionWorker(hardware=hardware)
    worker.start()
    try:
        worker.connect_hardware()
        wait_for(lambda: ("Cannot configure motor" if failure == "configure" else "short to ground")
                 in worker.output.read()[0].message)
        assert not worker.output.read()[0].armed
        assert not any(call[0] in {"enable", "move"} for call in hardware.calls)
        assert not hardware.state.enabled
    finally:
        worker.stop_worker()
        worker.join(timeout=2)


def test_setup_cancels_a_feedback_operation_during_its_stationary_baseline():
    hardware = FakeHardware()
    barriers = []
    def request(after):
        barriers.append(after)
        return len(barriers)
    worker = MotionWorker(pixel_config(), hardware=hardware, freshness_callback=request)
    worker.set_context(CONTEXT)
    worker.start()
    try:
        worker.connect_hardware()
        wait_for(lambda: worker.output.read()[0].connected)
        worker.apply_setup()
        wait_for(lambda: worker.output.read()[0].setup_applied and not worker.output.read()[0].armed)
        worker.arm()
        wait_for(lambda: worker.output.read()[0].armed)
        worker.align()
        wait_for(lambda: worker.output.read()[0].phase == "aligning")
        worker.apply_setup()
        wait_for(lambda: not worker.output.read()[0].armed)
        assert worker.controller._operation is None
        worker.submit_feedback(feedback("late", 10, barriers[-1] + .01))
        threading.Event().wait(.05)
        assert not any(call[0] == "move" for call in hardware.calls)
        assert not hardware.state.enabled
    finally:
        worker.stop_worker()
        worker.join(timeout=2)


def test_edit_current_and_apply_in_one_worker_batch_applies_successfully():
    hardware = FakeHardware()
    worker = MotionWorker(hardware=hardware)
    worker.connect_hardware()
    revision = worker.configure(MotionConfig(motor_current_ma=900))
    worker.apply_setup()
    worker.start()
    try:
        wait_for(lambda: worker.output.read()[0].setup_applied)
        assert worker.output.read()[0].config_revision == revision
        assert hardware.state.motor_current_ma == 900
        assert not hardware.state.enabled
    finally:
        worker.stop_worker()
        worker.join(timeout=2)


class BlockingHardware(FakeHardware):
    def __init__(self):
        super().__init__()
        self.block = False
        self.entered = threading.Event()
        self.release = threading.Event()

    def read_state(self):
        if self.block:
            self.block = False
            self.entered.set()
            if not self.release.wait(1):
                raise RuntimeError("Test did not release blocked SDK request")
        return super().read_state()


def test_stop_during_blocked_sdk_read_prevents_a_subsequent_nudge():
    hardware = BlockingHardware()
    worker = MotionWorker(MotionConfig(motor_current_ma=800), hardware=hardware)
    worker.start()
    try:
        worker.connect_hardware()
        wait_for(lambda: worker.output.read()[0].connected)
        worker.apply_setup()
        wait_for(lambda: worker.output.read()[0].setup_applied and not worker.output.read()[0].armed)
        worker.arm()
        wait_for(lambda: worker.output.read()[0].armed)
        hardware.block = True
        assert hardware.entered.wait(1)
        worker.set_keyboard(True, False)
        worker.stop()
        hardware.release.set()
        wait_for(lambda: "Motion stopped" in worker.output.read()[0].message)
        assert not any(call[0] == "move" for call in hardware.calls)
    finally:
        hardware.release.set()
        worker.stop_worker()
        worker.join(timeout=2)


def test_release_during_blocked_sdk_read_cancels_an_overdue_repeat():
    hardware = BlockingHardware()
    worker = MotionWorker(MotionConfig(motor_current_ma=800, repeat_hz=10), hardware=hardware)
    worker.start()
    try:
        worker.connect_hardware()
        wait_for(lambda: worker.output.read()[0].connected)
        worker.apply_setup()
        wait_for(lambda: worker.output.read()[0].setup_applied and not worker.output.read()[0].armed)
        worker.arm()
        wait_for(lambda: worker.output.read()[0].armed)
        worker.set_keyboard(True, False)
        wait_for(lambda: any(call[0] == "move" for call in hardware.calls))
        hardware.block = True
        assert hardware.entered.wait(1)
        threading.Event().wait(.12)
        worker.set_keyboard(False, False)
        hardware.release.set()
        threading.Event().wait(.15)
        assert len([call for call in hardware.calls if call[0] == "move"]) == 1
    finally:
        hardware.release.set()
        worker.stop_worker()
        worker.join(timeout=2)
