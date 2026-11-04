"""Numerical behavior and reference ownership, independent of a camera or UI."""
from dataclasses import replace

import cv2
import numpy as np
import pytest

from measure_shift.engine import MeasurementEngine, global_displacement, prepare_inputs
from measure_shift.models import FramePacket, MatchConfig, ReferenceSnapshot, Region


def texture(shape=(192, 256), seed=7):
    rng = np.random.default_rng(seed)
    image = rng.integers(15, 240, shape, dtype=np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0)


def translate(image, dx, dy):
    return cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]), image.shape[::-1])


def measure(reference, live, config=None):
    engine = MeasurementEngine()
    engine.set_reference(FramePacket.from_array(reference))
    return engine, engine.measure(FramePacket.from_array(live), config or MatchConfig())


@pytest.mark.parametrize("shift", [(0, 0), (8, -6), (-11, 9), (3.4, -2.25)])
def test_phase_signed_fractional_and_zero(shift):
    image = texture()
    _, result = measure(image, translate(image, *shift))
    assert result.usable, result.reasons
    assert (result.dx, result.dy) == pytest.approx(shift, abs=0.3)
    assert result.correction == (-result.dx, -result.dy)


def test_repeated_optimal_dft_calls_do_not_rewindow_cached_reference():
    image = texture((128, 128))
    engine = MeasurementEngine()
    engine.set_reference(FramePacket.from_array(image))
    config = MatchConfig()
    frame = FramePacket.from_array(translate(image, 4, -3))
    first = engine.measure(frame, config)
    prepared = engine._cache["prepared_reference"].copy()
    window = engine._cache["window"].copy()
    for _ in range(5):
        repeated = engine.measure(frame, config)
        assert (repeated.dx, repeated.dy, repeated.response) == pytest.approx((first.dx, first.dy, first.response), abs=1e-7)
    np.testing.assert_array_equal(engine._cache["prepared_reference"], prepared)
    np.testing.assert_array_equal(engine._cache["window"], window)


@pytest.mark.parametrize("shape", [(125, 135), (98, 134), (135, 135)])
def test_odd_optimal_dft_dimensions_have_no_half_pixel_zero_bias(shape):
    image = texture(shape)
    _, result = measure(image, image)
    assert result.usable, result.reasons
    assert (result.dx, result.dy) == pytest.approx((0, 0), abs=0.02)


def test_reference_owned_and_supplied_snapshot_preserved():
    image = texture()
    mutable_packet = FramePacket(image)
    engine = MeasurementEngine()
    snap = engine.set_reference(mutable_packet)
    image[:] = 0
    assert snap.frame.pixels.std() > 0
    assert not snap.frame.pixels.flags.writeable
    supplied = ReferenceSnapshot(FramePacket.from_array(texture()))
    assert engine.set_reference(supplied) is supplied
    engine.clear_reference()
    assert engine.reference is None
    assert not engine.measure(FramePacket.from_array(texture()), MatchConfig()).usable


@pytest.mark.parametrize("scale", [1.0, 0.5, 0.25])
def test_odd_roi_dimensions_scale_back_to_native_pixels(scale):
    image = texture((263, 357))
    config = MatchConfig(region=Region("rectangle", 39, 27, 269, 197), scale=scale)
    _, result = measure(image, translate(image, 12, -8), config)
    assert result.usable, result.reasons
    assert (result.dx, result.dy) == pytest.approx((12, -8), abs=0.8)
    inputs = prepare_inputs(image, image, config)
    assert inputs["scale_x"] == 269 / round(269 * scale)
    assert inputs["scale_y"] == 197 / round(197 * scale)
    assert inputs["reference_origin"] == (39, 27)


def test_global_displacement_retains_independently_centered_crop_origins():
    assert global_displacement((0.5, -2), (100, 60), (120, 90), 2, 4) == (21, 22)


def test_independent_live_crop_preserves_native_motion_and_reference_cache():
    image = texture((256, 320))
    live = translate(image, 50, 25)
    config = MatchConfig(region=Region("rectangle", 30, 30, 128, 128), scale=0.5)
    engine = MeasurementEngine()
    engine.set_reference(FramePacket.from_array(image))
    engine.measure(FramePacket.from_array(live), config)
    cached = engine._cache
    tracked = replace(config, live_region=Region("rectangle", 80, 55, 128, 128))
    result = engine.measure(FramePacket.from_array(live), tracked)
    assert result.usable, result.reasons
    assert (result.dx, result.dy) == pytest.approx((50, 25), abs=0.1)
    assert engine._cache is cached
    assert result.config_id == config.config_id
    diagnostic = prepare_inputs(image, live, tracked)
    assert diagnostic["reference_origin"] == (30, 30)
    assert diagnostic["live_origin"] == (80, 55)
    np.testing.assert_array_equal(diagnostic["reference"], diagnostic["live"])


def test_independent_live_crop_requires_equal_native_dimensions():
    image = texture((256, 320))
    config = MatchConfig(region=Region("rectangle", 30, 30, 128, 128),
                         live_region=Region("rectangle", 80, 55, 126, 128))
    _, result = measure(image, image, config)
    assert result.validity == "invalid"
    assert "equal native dimensions" in result.reasons[0]


def test_circle_excludes_outside_pixels_and_applies_only_circle_window():
    image = texture((192, 192))
    config = MatchConfig(region=Region("circle", 16, 16, 160, 160))
    inputs = prepare_inputs(image, image, config)
    np.testing.assert_allclose(inputs["prepared_reference"], inputs["centered_reference"] * inputs["window"])
    changed = image.copy()
    # Change only pixels beyond the circle, at least one Gaussian radius away
    # from every pixel that has a nonzero correlation weight.
    yy, xx = np.indices((160, 160))
    outside = (xx - 79) ** 2 + (yy - 79) ** 2 > 84 ** 2
    changed[16:176, 16:176][outside] = 255
    second = prepare_inputs(changed, changed, config)
    np.testing.assert_array_equal(inputs["prepared_reference"], second["prepared_reference"])
    _, result = measure(image, translate(image, -5, 7), config)
    assert result.usable, result.reasons
    assert (result.dx, result.dy) == pytest.approx((-5, 7), abs=0.3)


@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_template_roi_search_converts_origin_and_scale(scale):
    image = texture((263, 357))
    config = MatchConfig(method="template", region=Region("rectangle", 71, 55, 81, 73), scale=scale)
    _, result = measure(image, translate(image, 14, -10), config)
    assert result.usable, result.reasons
    assert (result.dx, result.dy) == pytest.approx((14, -10), abs=1.3)
    assert result.response > 0.8


def test_textureless_and_geometry_changes_are_invalid():
    _, result = measure(np.full((64, 64), 100, np.uint8), np.full((64, 64), 100, np.uint8))
    assert result.validity == "invalid"
    assert "texture" in result.reasons[0]
    _, mismatch = measure(texture((64, 64)), texture((65, 64)))
    assert mismatch.validity == "invalid"
    assert "geometry" in mismatch.reasons[0]


def test_low_overlap_marks_estimate_uncertain():
    image = texture()
    _, result = measure(image, translate(image, 60, 0), MatchConfig(min_overlap=0.95))
    assert result.validity == "uncertain"
    assert result.overlap < 0.95
    assert any("overlap" in reason for reason in result.reasons)


def test_same_overlap_residual_supports_correct_alignment():
    image = texture()
    _, result = measure(image, translate(image, 12, -7))
    assert result.residual_after < result.residual_before * 0.2
    assert 0.8 < result.overlap < 1


def test_strong_response_does_not_override_clearly_worse_alignment(monkeypatch):
    image = texture()
    monkeypatch.setattr(cv2, "phaseCorrelate", lambda *args: ((35.0, 20.0), 0.95))
    _, result = measure(image, image)
    assert result.response == 0.95
    assert result.validity == "uncertain"
    assert any("increases" in reason for reason in result.reasons)


def test_cache_configuration_content_changes_even_with_same_id():
    image = texture()
    engine = MeasurementEngine()
    engine.set_reference(FramePacket.from_array(image))
    config = MatchConfig()
    frame = FramePacket.from_array(image)
    engine.measure(frame, config)
    engine.measure(frame, replace(config, region=Region("rectangle", 20, 30, 100, 100)))
    assert engine._cache["prepared_reference"].shape == (100, 100)


def test_normal_measurement_does_not_retain_walkthrough_intermediates():
    image = texture()
    engine, result = measure(image, image)
    assert result.usable
    assert "reference" not in engine._cache
    assert "centered_reference" not in engine._cache


def test_invalid_config_and_template_nonrectangle_return_invalid():
    image = texture()
    for config in [MatchConfig(method="unknown"), MatchConfig(blur_size=4), MatchConfig(method="template")]:
        _, result = measure(image, image, config)
        assert result.validity == "invalid"
        assert result.reasons
