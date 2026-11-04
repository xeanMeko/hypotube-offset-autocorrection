"""On-demand diagnostics for an immutable, internally consistent image pair.

There is deliberately no scheduler here: callers decide when to spend resources.
Every branch computes only the requested tab, and never changes measurement data.
"""
from __future__ import annotations

import math
import time
from typing import Any

import cv2
import numpy as np

from .engine import phase_dft_shape, prepare_inputs
from .models import DiagnosticSnapshot, InspectionPair


TABS = ("Image", "Preprocessing", "Fourier", "Correlation", "Alignment", "Performance", "Slot β")


def _red_cyan(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    return np.stack((reference, current, current), axis=-1)


def _alignment(reference: np.ndarray, current: np.ndarray, dx: float, dy: float) -> tuple[dict[str, np.ndarray], np.ndarray]:
    height, width = reference.shape
    matrix = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
    aligned = cv2.warpAffine(current, matrix, (width, height), flags=cv2.INTER_LINEAR)
    support = cv2.warpAffine(np.ones(current.shape, np.float32), matrix, (width, height), flags=cv2.INTER_LINEAR)
    valid = support >= 1 - 1e-6
    before = cv2.absdiff(reference, current)
    after = cv2.absdiff(reference, aligned)
    after[~valid] = 0
    compensated = _red_cyan(reference, aligned)
    compensated[~valid] = 0
    return {
        "before_red_cyan": _red_cyan(reference, current),
        "compensated_red_cyan": compensated,
        "difference_before": before,
        "difference_after_valid_overlap": after,
    }, valid


def analyze_pair(pair: InspectionPair, tab: str) -> DiagnosticSnapshot:
    """Inspect exactly one tab without modifying its captured images or result.

    Arrays are owned by the snapshot and read-only. The correlation map is an
    independent reconstruction; OpenCV's measured displacement remains explicit.
    """
    if tab not in TABS:
        raise ValueError(f"Unknown analytics tab: {tab}")
    if (pair.result.reference_id, pair.result.frame_id, pair.result.config_id) != pair.key:
        raise ValueError("Diagnostic images, settings, and measurement must have matching provenance")
    started = time.perf_counter()
    reference, current = pair.reference.frame.pixels, pair.frame.pixels
    arrays: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {
        "reference_id": pair.reference.reference_id,
        "frame_id": pair.frame.frame_id,
        "config_id": pair.config.config_id,
        "measured_dx_px": pair.result.dx,
        "measured_dy_px": pair.result.dy,
        "opencv_response": pair.result.response,
        "measurement_validity": pair.result.validity,
    }
    if tab == "Image":
        arrays.update(reference=reference.copy(), live=current.copy())
        for name, pixels in (("reference", reference), ("live", current)):
            arrays[f"histogram_{name}"] = np.bincount(pixels.ravel(), minlength=256)
            arrays[f"center_line_{name}"] = pixels[pixels.shape[0] // 2].copy()
        metrics.update(histogram_x="8-bit intensity", profile_x="native horizontal pixels", profile="Center row; draw a line in the image view for a custom profile")
    elif tab in {"Preprocessing", "Fourier", "Correlation"}:
        inputs = prepare_inputs(reference, current, pair.config)
        metrics.update(
            native_px_per_analysis_x=float(inputs["scale_x"]),
            native_px_per_analysis_y=float(inputs["scale_y"]),
            reference_origin=list(inputs["reference_origin"]),
            live_origin=list(inputs["live_origin"]),
            analysis_resolution="Native" if pair.config.scale == 1 else f"Reduced ({pair.config.scale:g}×)",
        )
        if tab == "Preprocessing":
            for name in ("reference", "live", "blur_reference", "blur_live", "centered_reference", "centered_live", "window", "prepared_reference", "prepared_live"):
                arrays[name] = inputs[name].copy()
            metrics["window"] = "Applied once to centered inputs" if pair.config.method == "phase" else "Template matching: no taper"
        elif tab == "Fourier":
            for name in ("reference", "live"):
                plane = inputs[f"prepared_{name}"]
                fft_shape = phase_dft_shape(plane.shape) if pair.config.method == "phase" else plane.shape
                spectrum = np.fft.fftshift(np.fft.fft2(plane, s=fft_shape))
                arrays[f"log_magnitude_{name}"] = np.log1p(np.abs(spectrum)).astype(np.float32)
                arrays[f"phase_{name}"] = np.angle(spectrum).astype(np.float32)
            if inputs["prepared_reference"].shape == inputs["prepared_live"].shape:
                fft_shape = phase_dft_shape(inputs["prepared_reference"].shape) if pair.config.method == "phase" else inputs["prepared_reference"].shape
                cross = np.fft.fft2(inputs["prepared_live"], s=fft_shape) * np.conj(np.fft.fft2(inputs["prepared_reference"], s=fft_shape))
                arrays["normalized_cross_phase"] = np.fft.fftshift(np.angle(cross)).astype(np.float32)
                metrics["phase_units"] = "radians"
            else:
                metrics["cross_phase"] = "Unavailable for unequal template and search-image shapes"
        elif tab == "Correlation":
            _correlation(inputs, pair, arrays, metrics)
    elif tab == "Alignment":
        if reference.shape != current.shape:
            metrics["alignment"] = "Images have different geometry; capture a new reference"
        elif not (math.isfinite(pair.result.dx) and math.isfinite(pair.result.dy)):
            metrics["alignment"] = "No finite displacement available"
            arrays["before_red_cyan"] = _red_cyan(reference, current)
        else:
            alignment_arrays, valid = _alignment(reference, current, pair.result.dx, pair.result.dy)
            arrays.update(alignment_arrays)
            metrics["valid_overlap_fraction"] = float(np.mean(valid))
            metrics["residual_before_mae"] = float(np.mean(arrays["difference_before"][valid])) if valid.any() else None
            metrics["residual_after_mae"] = float(np.mean(arrays["difference_after_valid_overlap"][valid])) if valid.any() else None
            metrics["residual_area"] = "Native full-frame valid overlap; measurement quality may use a selected ROI"
            metrics["measurement_residual_before"] = pair.result.residual_before
            metrics["measurement_residual_after"] = pair.result.residual_after
            metrics["preview"] = "Software compensated using negative measured displacement"
    elif tab == "Performance":
        metrics.update(pair.runtime_metrics)
        metrics.update({f"measurement_{name}": float(value) for name, value in pair.result.timings.items()})
        metrics["frame_age_at_measurement_ms"] = max(0.0, (pair.result.measured_at - pair.frame.received_at) * 1000)
        metrics["source_wait_read_ms"] = pair.frame.source_wait_read_ms
        metrics["camera_source"] = pair.frame.source_name
        metrics["native_width"] = current.shape[1]
        metrics["native_height"] = current.shape[0]
    elif tab == "Slot β":
        from .slots import detect_slots

        candidates = detect_slots(current)
        overlay = cv2.cvtColor(current, cv2.COLOR_GRAY2RGB)
        for candidate in candidates:
            x, y, width, height = candidate.bounds
            cv2.rectangle(overlay, (x, y), (x + width - 1, y + height - 1), (255, 190, 0), 2)
            tip = tuple(int(round(v)) for v in candidate.tip)
            cv2.circle(overlay, tip, 4, (255, 40, 80), -1)
            cv2.putText(overlay, f"Candidate {candidate.id}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 190, 0), 1)
        runtime = pair.runtime_metrics
        status = runtime.get("slot_status", "Off")
        metrics.update(
            slot_status=status,
            slot_id=runtime.get("slot_id"),
            reference_tip_px=runtime.get("reference_tip_px"),
            live_tip_px=runtime.get("live_tip_px"),
            selected_reference_region_px=list(pair.config.region.bounds(reference.shape)),
            selected_live_region_px=list(pair.config.live_region.bounds(current.shape)) if pair.config.live_region is not None else None,
            proposal_identity="New candidates are proposals; the captured selected slot identity is unchanged",
        )
        if status == "Tracking" and pair.config.live_region is not None:
            x, y, width, height = pair.config.live_region.bounds(current.shape)
            selected_color = (70, 220, 140)
            cv2.rectangle(overlay, (x, y), (x + width - 1, y + height - 1), selected_color, 2)
            cv2.putText(overlay, f"Selected slot {runtime.get('slot_id', '?')}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, selected_color, 1)
            live_tip = runtime.get("live_tip_px")
            if live_tip is not None and len(live_tip) == 2 and np.isfinite(live_tip).all():
                tip = tuple(int(round(value)) for value in live_tip)
                cv2.drawMarker(overlay, tip, selected_color, cv2.MARKER_CROSS, 13, 2)
        arrays["slot_candidates_beta"] = overlay
        metrics["candidate_count"] = len(candidates)
        metrics["candidate_scores"] = [round(candidate.score, 4) for candidate in candidates]
        metrics["detector"] = "Experimental; select an identity to track in the main view"
    metrics["analytics_ms"] = (time.perf_counter() - started) * 1000
    for value in arrays.values():
        value.setflags(write=False)
    return DiagnosticSnapshot(pair, tab, arrays, metrics)


def _correlation(inputs: dict, pair: InspectionPair, arrays: dict, metrics: dict) -> None:
    reference = inputs["prepared_reference"]
    current = inputs["prepared_live"]
    sx, sy = inputs["scale_x"], inputs["scale_y"]
    rx, ry = inputs["reference_origin"]
    lx, ly = inputs["live_origin"]
    if pair.config.method == "template":
        correlation = cv2.matchTemplate(inputs["blur_live"], inputs["blur_reference"], cv2.TM_CCOEFF_NORMED)
        axes_x = np.arange(correlation.shape[1]) * sx + lx - rx
        axes_y = np.arange(correlation.shape[0]) * sy + ly - ry
        metrics["map_label"] = "Reconstructed normalized template correlation; measured result shown separately"
    else:
        fft_shape = phase_dft_shape(reference.shape)
        cross = np.fft.fft2(current, s=fft_shape) * np.conj(np.fft.fft2(reference, s=fft_shape))
        magnitude = np.abs(cross)
        normalized = np.divide(cross, magnitude, out=np.zeros_like(cross), where=magnitude > np.finfo(float).eps)
        correlation = np.fft.fftshift(np.fft.ifft2(normalized).real).astype(np.float32)
        height, width = correlation.shape
        axes_x = (np.arange(width) - width // 2) * sx + lx - rx
        axes_y = (np.arange(height) - height // 2) * sy + ly - ry
        metrics["map_label"] = "Reconstructed signed phase-correlation map; independent from OpenCV subpixel estimator"
    iy, ix = np.unravel_index(np.argmax(correlation), correlation.shape)
    radius = 12
    left, top = max(0, ix - radius), max(0, iy - radius)
    right, bottom = min(correlation.shape[1], ix + radius + 1), min(correlation.shape[0], iy + radius + 1)
    arrays["reconstructed_correlation"] = correlation
    arrays["peak_zoom"] = correlation[top:bottom, left:right].copy()
    arrays["correlation_horizontal"] = correlation[iy].copy()
    arrays["correlation_vertical"] = correlation[:, ix].copy()
    metrics.update(
        reconstructed_integer_peak_dx_px=float(axes_x[ix]),
        reconstructed_integer_peak_dy_px=float(axes_y[iy]),
        reconstructed_peak_value=float(correlation[iy, ix]),
        slice_x_native_px=axes_x.tolist(),
        slice_y_native_px=axes_y.tolist(),
        correlation_rect_native_px=[float(axes_x[0] - sx / 2), float(axes_y[0] - sy / 2), float(correlation.shape[1] * sx), float(correlation.shape[0] * sy)],
        peak_zoom_rect_native_px=[float(axes_x[left] - sx / 2), float(axes_y[top] - sy / 2), float((right - left) * sx), float((bottom - top) * sy)],
        peak_zoom_x_native_px=axes_x[left:right].tolist(),
        peak_zoom_y_native_px=axes_y[top:bottom].tolist(),
    )
