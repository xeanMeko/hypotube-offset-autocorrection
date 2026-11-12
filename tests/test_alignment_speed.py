"""Calibrated corrections use remaining X distance without a learning ramp."""

import pytest

from test_adaptive_alignment import Axis, CAMERA, calibrated


@pytest.mark.parametrize("gain", [.05, -.05])
def test_small_pixel_response_uses_large_moves_immediately(gain):
    axis = Axis(config=calibrated(scale=gain, align_max_steps=100),
                error=80, gain=gain)

    axis.run()

    assert "X aligned" in axis.controller.message
    assert abs(axis.error) <= CAMERA.tolerance_px
    assert abs(axis.moves[0]) == 100
    assert len(axis.moves) <= 23
    assert max(map(abs, axis.moves)) <= 100
    assert axis.calibration_captures == 0
    assert axis.controller.config.pixels_per_step == gain


@pytest.mark.parametrize("gain", [1, -1])
def test_session_scale_corrects_large_dx_without_ramping_from_one_step(gain):
    axis = Axis(config=calibrated(scale=gain, align_max_steps=100),
                error=800, gain=gain)

    axis.run()

    assert "X aligned" in axis.controller.message
    assert abs(axis.error) <= CAMERA.tolerance_px
    assert abs(axis.moves[0]) == 100
    assert len(axis.moves) <= 14
    assert max(map(abs, axis.moves)) <= 100
    assert abs(axis.moves[-1]) < max(map(abs, axis.moves))


@pytest.mark.parametrize("gain", [2.5, -2.5])
@pytest.mark.parametrize("error", [20, -20])
def test_large_move_allowance_still_shrinks_corrections_near_the_target(gain, error):
    axis = Axis(config=calibrated(scale=gain, align_max_steps=100),
                error=error, gain=gain)

    axis.run()

    assert "X aligned" in axis.controller.message
    assert abs(axis.error) <= CAMERA.tolerance_px
    assert abs(axis.moves[0]) == 6
    assert abs(axis.moves[-1]) == 1
    assert max(map(abs, axis.moves)) <= 6


@pytest.mark.parametrize("gain", [1, -1])
@pytest.mark.parametrize("error", [2.4, -2.4])
def test_known_scale_keeps_a_small_initial_dx_correction_small(gain, error):
    axis = Axis(config=calibrated(scale=gain, align_max_steps=100),
                error=error, gain=gain)

    axis.run()

    assert "X aligned" in axis.controller.message
    assert abs(axis.error) <= CAMERA.tolerance_px
    assert sum(map(abs, axis.moves)) <= 2


@pytest.mark.parametrize("dy", [-10_000, 10_000])
def test_valid_vertical_offset_does_not_change_calibrated_x_moves(dy):
    config = calibrated(scale=.25, align_max_steps=100)
    baseline = Axis(config=config, error=80, gain=.25)
    baseline.run()
    axis = Axis(config=config, error=80, gain=.25)

    for _ in range(150):
        if axis.controller._operation is None:
            break
        axis.observe(dy=dy)
        if axis.pending_move is not None:
            axis.finish()
    else:
        pytest.fail("Alignment with a vertical offset did not finish")

    assert "X aligned" in axis.controller.message
    assert abs(axis.error) <= CAMERA.tolerance_px
    assert axis.moves == baseline.moves


@pytest.mark.parametrize("limit", [1, 3, 7])
@pytest.mark.parametrize("gain", [1, -1])
def test_calibrated_x_moves_keep_the_configured_user_limit(limit, gain):
    axis = Axis(config=calibrated(scale=gain, align_max_steps=limit),
                error=30, gain=gain)

    axis.run()

    assert "X aligned" in axis.controller.message
    assert abs(axis.error) <= CAMERA.tolerance_px
    assert abs(axis.moves[0]) == limit
    assert max(map(abs, axis.moves)) <= limit
    assert axis.controller.config.align_max_steps == limit


@pytest.mark.parametrize("gain", [.5, -.5])
def test_accepted_uncertain_matches_use_session_scale_for_larger_corrections(gain):
    axis = Axis(config=calibrated(scale=gain, align_max_steps=100),
                error=80, gain=gain)

    axis.observe(usable=False, validity="uncertain")

    assert abs(axis.pending_move.steps) == 100
    assert axis.calibration_captures == 0
    axis.finish()
    axis.observe()
    assert abs(axis.pending_move.steps) > 1
    axis.finish()
    axis.run()
    assert "X aligned" in axis.controller.message


@pytest.mark.parametrize("scale", [1, -1, .05, -.05])
def test_error_growth_keeps_the_next_calibrated_correction_within_user_limit(scale):
    axis = Axis(config=calibrated(scale=scale, align_max_steps=100),
                error=80, gain=-scale)
    axis.observe()
    axis.finish()

    axis.observe()

    assert axis.error > 80
    assert axis.pending_move is not None
    assert axis.controller.armed
    assert 1 < abs(axis.pending_move.steps) <= 100
    assert axis.controller.config.pixels_per_step == scale
    assert axis.calibration_captures == 0
