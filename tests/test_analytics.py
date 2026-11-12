"""Diagnostic semantics, provenance, laziness, and measurement independence."""
from dataclasses import replace

import cv2
import numpy as np
import pytest

from measure_shift import analytics
from measure_shift.analytics import TABS, analyze_pair
from measure_shift.engine import MeasurementEngine, phase_dft_shape
from measure_shift.models import FramePacket, InspectionPair, MatchConfig, Region


def pair_for_shift(dx=8, dy=-6, config=None, shape=(192, 256)):
    rng = np.random.default_rng(92)
    reference = cv2.GaussianBlur(rng.integers(20, 235, shape, dtype=np.uint8), (3, 3), 0)
    live = cv2.warpAffine(reference, np.float32([[1, 0, dx], [0, 1, dy]]), shape[::-1])
    engine = MeasurementEngine()
    snapshot = engine.set_reference(FramePacket.from_array(reference))
    frame = FramePacket.from_array(live)
    config = config or MatchConfig()
    result = engine.measure(frame, config)
    return engine, InspectionPair(snapshot, frame, config, result)


def test_every_tab_preserves_images_cache_and_measurement():
    engine, pair = pair_for_shift()
    cached = engine._cache["prepared_reference"].copy()
    before = pair.reference.frame.pixels.copy(), pair.frame.pixels.copy()
    for tab in TABS:
        diagnostic = analyze_pair(pair, tab)
        assert diagnostic.pair is pair
        assert diagnostic.tab == tab
        assert all(not a.flags.writeable for a in diagnostic.arrays.values())
    repeated = engine.measure(pair.frame, pair.config)
    assert (repeated.dx, repeated.dy, repeated.response) == pytest.approx((pair.result.dx, pair.result.dy, pair.result.response), abs=1e-7)
    np.testing.assert_array_equal(cached, engine._cache["prepared_reference"])
    np.testing.assert_array_equal(pair.reference.frame.pixels, before[0])
    np.testing.assert_array_equal(pair.frame.pixels, before[1])


@pytest.mark.parametrize("tab", ["Image", "Alignment", "Performance"])
def test_non_transform_tabs_do_not_prepare_or_fft(monkeypatch, tab):
    _, pair = pair_for_shift()

    def unexpected(*args, **kwargs):
        raise AssertionError("Unrequested preprocessing or Fourier work")

    monkeypatch.setattr(analytics, "prepare_inputs", unexpected)
    monkeypatch.setattr(np.fft, "fft2", unexpected)
    analyze_pair(pair, tab)


@pytest.mark.parametrize("shape", [(192, 256), (125, 135)])
def test_reconstructed_signed_peak_and_slice_axes(shape):
    _, pair = pair_for_shift(8, -6, shape=shape)
    diagnostic = analyze_pair(pair, "Correlation")
    assert "Reconstructed" in diagnostic.metrics["map_label"]
    assert diagnostic.metrics["reconstructed_integer_peak_dx_px"] == pytest.approx(8, abs=1)
    assert diagnostic.metrics["reconstructed_integer_peak_dy_px"] == pytest.approx(-6, abs=1)
    assert diagnostic.arrays["reconstructed_correlation"].shape == phase_dft_shape(shape)
    assert len(diagnostic.metrics["slice_x_native_px"]) == len(diagnostic.arrays["correlation_horizontal"])
    assert diagnostic.metrics["measured_dx_px"] == pair.result.dx


def test_correlation_axes_use_actual_native_scale():
    config = MatchConfig(region=Region("rectangle", 39, 27, 269, 197), scale=0.5)
    _, pair = pair_for_shift(12, -8, config, (263, 357))
    diagnostic = analyze_pair(pair, "Correlation")
    assert diagnostic.metrics["reconstructed_integer_peak_dx_px"] == pytest.approx(12, abs=1)
    assert diagnostic.metrics["reconstructed_integer_peak_dy_px"] == pytest.approx(-8, abs=1)
    x = diagnostic.metrics["slice_x_native_px"]
    assert x[1] - x[0] == pytest.approx(269 / round(269 * 0.5))


def test_independently_recentered_live_crop_preserves_original_displacement():
    config = MatchConfig(
        region=Region("rectangle", 39, 27, 160, 128),
        live_region=Region("rectangle", 51, 19, 160, 128),
        scale=0.5,
    )
    _, pair = pair_for_shift(12, -8, config)
    diagnostic = analyze_pair(pair, "Correlation")
    assert diagnostic.metrics["reconstructed_integer_peak_dx_px"] == 12
    assert diagnostic.metrics["reconstructed_integer_peak_dy_px"] == -8
    assert diagnostic.metrics["reference_origin"] == [39, 27]
    assert diagnostic.metrics["live_origin"] == [51, 19]
    rect = diagnostic.metrics["correlation_rect_native_px"]
    assert rect[0] + 1 == diagnostic.metrics["slice_x_native_px"][0]
    assert rect[1] + 1 == diagnostic.metrics["slice_y_native_px"][0]
    zoom = diagnostic.metrics["peak_zoom_rect_native_px"]
    assert zoom[0] + 1 == diagnostic.metrics["peak_zoom_x_native_px"][0]
    assert zoom[1] + 1 == diagnostic.metrics["peak_zoom_y_native_px"][0]


def test_compensation_uses_negative_shift_and_full_interpolation_support():
    _, pair = pair_for_shift(8, -6)
    # Use the exact known shift to isolate preview semantics from subpixel fitting.
    pair = replace(pair, result=replace(pair.result, dx=8, dy=-6))
    diagnostic = analyze_pair(pair, "Alignment")
    assert set(diagnostic.arrays) == {
        "before_red_cyan", "compensated_red_cyan",
        "difference_before", "difference_after_valid_overlap",
    }
    valid = np.zeros(pair.frame.pixels.shape, dtype=bool)
    valid[6:, :-8] = True
    assert diagnostic.metrics["valid_overlap_fraction"] == pytest.approx(valid.mean())
    assert not diagnostic.arrays["compensated_red_cyan"][~valid].any()
    assert not diagnostic.arrays["difference_after_valid_overlap"][~valid].any()
    assert diagnostic.metrics["residual_after_mae"] == 0
    assert diagnostic.metrics["residual_before_mae"] > 1
    # A fractional edge has partial interpolation support and must be excluded.
    pair = replace(pair, result=replace(pair.result, dx=0.5, dy=0))
    fraction = analyze_pair(pair, "Alignment")
    assert fraction.metrics["valid_overlap_fraction"] == pytest.approx(255 / 256)
    assert not fraction.arrays["compensated_red_cyan"][:, -1].any()
    assert not fraction.arrays["difference_after_valid_overlap"][:, -1].any()


def test_pair_provenance_rejects_mixed_measurements():
    _, pair = pair_for_shift()
    with pytest.raises(ValueError, match="provenance"):
        analyze_pair(replace(pair, result=replace(pair.result, config_id="outdated")), "Image")
    with pytest.raises(ValueError, match="Unknown"):
        analyze_pair(pair, "not-a-tab")


def test_image_histograms_and_performance_units():
    _, pair = pair_for_shift()
    image = analyze_pair(pair, "Image")
    assert image.arrays["histogram_reference"].sum() == pair.reference.frame.pixels.size
    performance = analyze_pair(pair, "Performance")
    assert performance.metrics["measurement_total_ms"] == pair.result.timings["total_ms"]


def test_performance_uses_captured_rates_and_keeps_a_detached_metric_mapping():
    _, pair = pair_for_shift()
    rates = {"camera_fps": 45.0, "measurement_fps": 38.0, "frame_age_at_display_ms": 24.0}
    pair = replace(pair, runtime_metrics=rates.copy())
    performance = analyze_pair(pair, "Performance")
    assert all(performance.metrics[key] == value for key, value in rates.items())
    # Diagnostics own their scalar metrics, while a different live receipt can
    # carry new rates without changing the inspected pair's captured values.
    newer = replace(pair, runtime_metrics={"camera_fps": 12.0})
    analyze_pair(newer, "Performance")
    assert performance.metrics["camera_fps"] == 45.0
    performance.metrics["camera_fps"] = 99
    assert pair.runtime_metrics["camera_fps"] == 45.0
