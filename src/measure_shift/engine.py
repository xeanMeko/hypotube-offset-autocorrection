"""Translation measurement with immutable references and explicit coordinates."""
from __future__ import annotations

from dataclasses import replace
import math
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .models import FramePacket, MatchConfig, MatchResult, ReferenceSnapshot


def global_displacement(
    local_shift: tuple[float, float],
    reference_origin: tuple[float, float],
    live_origin: tuple[float, float],
    scale_x: float,
    scale_y: float,
) -> tuple[float, float]:
    """Convert a crop-local shift to native pixels (right/down are positive).

    Scales are native pixels per analysis sample, including actual resize
    rounding. Independently recentered crops must retain both native origins.
    """
    return (
        float(live_origin[0] - reference_origin[0] + local_shift[0] * scale_x),
        float(live_origin[1] - reference_origin[1] + local_shift[1] * scale_y),
    )


def _validate_image(image: np.ndarray) -> None:
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Inputs must be 8-bit grayscale images")
    if min(image.shape) < 8:
        raise ValueError("Images must be at least 8 × 8 pixels")


def _validate_config(config: MatchConfig) -> None:
    if config.method != "phase":
        raise ValueError("Only phase correlation is supported")
    if config.scale not in {1.0, 0.5, 0.25}:
        raise ValueError("Analysis scale must be 1, 1/2, or 1/4")
    if not isinstance(config.blur_size, int) or not 1 <= config.blur_size <= 31 or config.blur_size % 2 == 0:
        raise ValueError("Blur size must be an odd integer from 1 to 31")
    if not 0 <= config.min_response <= 1 or not 0 < config.min_overlap <= 1:
        raise ValueError("Invalid quality thresholds")
    if not math.isfinite(config.min_texture) or config.min_texture < 0:
        raise ValueError("Invalid texture threshold")


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if min(width, height) < 8:
        raise ValueError("Analysis region must be at least 8 × 8 pixels at the selected scale")
    if image.shape == (height, width):
        return np.ascontiguousarray(image).copy()
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def phase_dft_shape(shape: tuple[int, int]) -> tuple[int, int]:
    """Choose even optimal FFT dimensions to avoid odd-center half-pixel bias."""
    dimensions = []
    for size in shape:
        optimal = cv2.getOptimalDFTSize(size)
        while optimal % 2:
            optimal = cv2.getOptimalDFTSize(optimal + 1)
        dimensions.append(optimal)
    return dimensions[0], dimensions[1]


def _phase_canvas(prepared: np.ndarray) -> np.ndarray:
    height, width = phase_dft_shape(prepared.shape)
    if prepared.shape == (height, width):
        return prepared
    return cv2.copyMakeBorder(prepared, 0, height - prepared.shape[0],
                              0, width - prepared.shape[1], cv2.BORDER_CONSTANT)


def _circle_window(shape: tuple[int, int], blur_size: int) -> np.ndarray:
    """Cosine taper inside a circular/elliptical bounding box, with blur margin."""
    height, width = shape
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(mask, ((width - 1) // 2, (height - 1) // 2),
                (width // 2, height // 2), 0, 0, 360, 255, -1)
    # Pad before the distance transform so a disk touching the crop boundary
    # still has a real outside boundary there.
    distance = cv2.distanceTransform(np.pad(mask, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    margin = blur_size // 2 + 1
    taper = max(2.0, min(height, width) * 0.05)
    ramp = np.clip((distance - margin) / taper, 0.0, 1.0)
    window = (0.5 - 0.5 * np.cos(np.pi * ramp)).astype(np.float32)
    if float(window.sum()) < 10:
        raise ValueError("Circular region is too small for the selected blur and scale")
    return window


def _prepare_plane(raw: np.ndarray, config: MatchConfig, window: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    blurred = cv2.GaussianBlur(raw, (config.blur_size, config.blur_size), 0).astype(np.float32)
    if config.region.kind == "circle":
        mean = float(np.sum(blurred * window, dtype=np.float64) / window.sum(dtype=np.float64))
        texture = float(blurred[window > 0].std())
    else:
        mean = float(blurred.mean())
        texture = float(blurred.std())
    centered = blurred - mean
    # OpenCV can modify inputs when its window argument is supplied at an
    # optimal DFT size. Own this product and supply no window to phaseCorrelate.
    prepared = np.ascontiguousarray(centered * window)
    return blurred, centered, prepared, texture


def _prepare_reference(image: np.ndarray, config: MatchConfig, *, intermediates: bool = True) -> dict[str, Any]:
    _validate_image(image)
    _validate_config(config)
    x, y, width, height = config.region.bounds(image.shape)
    out_w, out_h = round(width * config.scale), round(height * config.scale)
    raw = _resize(image[y:y + height, x:x + width], out_w, out_h)
    scale_x, scale_y = width / out_w, height / out_h
    origin = (float(x), float(y))
    window = (_circle_window(raw.shape, config.blur_size)
              if config.region.kind == "circle"
              else cv2.createHanningWindow((out_w, out_h), cv2.CV_32F))
    live_size = (out_w, out_h)
    blurred, centered, prepared, texture = _prepare_plane(raw, config, window)
    result = {"blur_reference": blurred, "prepared_reference": prepared,
              "window": window, "reference_origin": origin,
              "scale_x": scale_x, "scale_y": scale_y,
              "texture_reference": texture, "native_shape": image.shape,
              "bounds": (x, y, width, height), "live_size": live_size}
    if intermediates:
        result.update(reference=raw, centered_reference=centered)
    if not intermediates:
        result["phase_reference"] = _phase_canvas(prepared)
    return result


def _prepare_live(image: np.ndarray, config: MatchConfig, reference: dict[str, Any], *, intermediates: bool = True) -> dict[str, Any]:
    _validate_image(image)
    if image.shape != reference["native_shape"]:
        raise ValueError("Image geometry changed; capture a new reference")
    x, y, width, height = reference["bounds"]
    live_region = getattr(config, "live_region", None)
    if live_region is not None:
        live_x, live_y, live_width, live_height = live_region.bounds(image.shape)
        if (live_width, live_height) != (width, height):
            raise ValueError("Reference and live analysis regions must have equal native dimensions")
        x, y = live_x, live_y
    raw = _resize(image[y:y + height, x:x + width], *reference["live_size"])
    origin = (float(x), float(y))
    window = reference["window"]
    blurred, centered, prepared, texture = _prepare_plane(raw, config, window)
    result = {"blur_live": blurred, "prepared_live": prepared,
              "live_origin": origin, "texture_live": texture}
    if intermediates:
        result.update(live=raw, centered_live=centered)
    return result


def prepare_inputs(reference: np.ndarray, live: np.ndarray, config: MatchConfig) -> dict[str, Any]:
    """Return owned analysis intermediates for a frozen diagnostic pair.

    Origins are native pixels; scale_x/y are native pixels per analysis sample.
    Phase arrays share a shape; optional independent crops retain their origins.
    """
    prepared = _prepare_reference(reference, config)
    return {**prepared, **_prepare_live(live, config, prepared)}


def _quality(inputs: dict[str, Any], local_shift: tuple[float, float], config: MatchConfig) -> tuple[float, float | None, float | None]:
    """Compare the same supported coordinates on a bounded-size image grid."""
    ref, live = inputs["blur_reference"], inputs["blur_live"]
    height, width = ref.shape
    reduction = min(1.0, 512.0 / max(height, width))
    out_w, out_h = max(1, round(width * reduction)), max(1, round(height * reduction))
    ratio_x, ratio_y = width / out_w, height / out_h
    ref_small = cv2.resize(ref, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    if config.region.kind == "circle":
        support = (inputs["window"] > 0).astype(np.float32)
    else:
        support = np.ones(ref.shape, dtype=np.float32)
    ref_valid = cv2.resize(support, (out_w, out_h), interpolation=cv2.INTER_LINEAR) >= 1 - 1e-6

    def warp(shift: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        transform = np.float32([[1 / ratio_x, 0, (0.5 - shift[0]) / ratio_x - 0.5],
                                [0, 1 / ratio_y, (0.5 - shift[1]) / ratio_y - 0.5]])
        image = cv2.warpAffine(live, transform, (out_w, out_h), flags=cv2.INTER_LINEAR)
        covered = cv2.warpAffine(support, transform, (out_w, out_h), flags=cv2.INTER_LINEAR)
        return image, covered >= 1 - 1e-6

    ref_origin, live_origin = inputs["reference_origin"], inputs["live_origin"]
    before_shift = ((ref_origin[0] - live_origin[0]) / inputs["scale_x"],
                    (ref_origin[1] - live_origin[1]) / inputs["scale_y"])
    before, before_valid = warp(before_shift)
    after, after_valid = warp(local_shift)
    overlap = float(np.count_nonzero(ref_valid & after_valid) / max(1, np.count_nonzero(ref_valid)))
    shared = ref_valid & before_valid & after_valid
    if np.count_nonzero(shared) < 16:
        return overlap, None, None
    before_error = float(np.abs(ref_small[shared] - before[shared]).mean())
    after_error = float(np.abs(ref_small[shared] - after[shared]).mean())
    return overlap, before_error, after_error


class MeasurementEngine:
    """Worker-owned engine; caches only one reference/configuration pairing."""

    def __init__(self) -> None:
        self._reference: ReferenceSnapshot | None = None
        self._cache_key: tuple[str, MatchConfig] | None = None
        self._cache: dict[str, Any] | None = None

    @property
    def reference(self) -> ReferenceSnapshot | None:
        return self._reference

    def set_reference(self, frame: FramePacket | ReferenceSnapshot) -> ReferenceSnapshot:
        if isinstance(frame, ReferenceSnapshot):
            _validate_image(frame.frame.pixels)
            snapshot = frame
        else:
            _validate_image(frame.pixels)
            owned = np.ascontiguousarray(frame.pixels).copy()
            owned.setflags(write=False)
            snapshot = ReferenceSnapshot(replace(frame, pixels=owned))
        self._reference = snapshot
        self._cache_key = None
        self._cache = None
        return snapshot

    def clear_reference(self) -> None:
        self._reference = None
        self._cache_key = None
        self._cache = None

    def measure(self, frame: FramePacket, config: MatchConfig) -> MatchResult:
        started = perf_counter()
        reference = self._reference
        provenance = {"frame_id": frame.frame_id,
                      "reference_id": reference.reference_id if reference else "",
                      "config_id": config.config_id}
        timings: dict[str, float] = {}
        if reference is None:
            return MatchResult(**provenance, reasons=("Capture a reference first",))
        try:
            _validate_config(config)
            reference_config = replace(config, live_region=None) if hasattr(config, "live_region") else config
            key = (reference.reference_id, reference_config)
            if key != self._cache_key:
                self._cache = _prepare_reference(reference.frame.pixels, config, intermediates=False)
                self._cache_key = key
            assert self._cache is not None
            inputs = {**self._cache, **_prepare_live(frame.pixels, config, self._cache, intermediates=False)}
            prepared_at = perf_counter()
            timings["preprocessing_ms"] = (prepared_at - started) * 1000
            # Tiny nonzero numerical noise is not useful texture even when a
            # user deliberately lowers the configured threshold to zero.
            if min(inputs["texture_reference"], inputs["texture_live"]) < max(config.min_texture, 1e-3):
                raise ValueError("Too little texture inside the analysis region")
            # Explicitly choose an even optimal canvas. OpenCV's centered
            # peak convention otherwise adds half a sample when an optimal
            # DFT dimension is odd, including some downsampled ROIs.
            local, response = cv2.phaseCorrelate(inputs["phase_reference"],
                                               _phase_canvas(inputs["prepared_live"]))
            matched_at = perf_counter()
            timings["matching_ms"] = (matched_at - prepared_at) * 1000
            dx, dy = global_displacement(local, inputs["reference_origin"], inputs["live_origin"],
                                         inputs["scale_x"], inputs["scale_y"])
            if not all(math.isfinite(value) for value in (dx, dy, response)):
                raise ValueError("No finite displacement estimate was found")
            overlap, before, after = _quality(inputs, local, config)
            reasons = []
            if response < config.min_response:
                reasons.append("Weak match response")
            if overlap < config.min_overlap:
                reasons.append("Too little overlapping image area")
            ended = perf_counter()
            timings["quality_ms"] = (ended - matched_at) * 1000
            timings["total_ms"] = (ended - started) * 1000
            return MatchResult(**provenance, dx=dx, dy=dy, response=float(response),
                               validity="uncertain" if reasons else "valid", reasons=tuple(reasons),
                               overlap=overlap, residual_before=before, residual_after=after, timings=timings)
        except (ValueError, TypeError, cv2.error) as error:
            timings["total_ms"] = (perf_counter() - started) * 1000
            return MatchResult(**provenance, reasons=(str(error),), timings=timings)
