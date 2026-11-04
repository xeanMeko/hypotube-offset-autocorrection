"""Portable exports: native images, numeric arrays, settings and provenance."""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
import tempfile
import zipfile

import cv2
import numpy as np

from .models import DiagnosticSnapshot, InspectionPair, MatchConfig


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def pair_metadata(pair: InspectionPair) -> dict:
    if (pair.result.reference_id, pair.result.frame_id, pair.result.config_id) != pair.key:
        raise ValueError("Images, settings, and measurement must have matching provenance")
    return json_safe({
        "format_version": 1,
        "reference_id": pair.reference.reference_id,
        "reference_frame_id": pair.reference.frame.frame_id,
        "reference_source": pair.reference.frame.source_name,
        "reference_captured_at": pair.reference.frame.captured_at,
        "frame_id": pair.frame.frame_id,
        "source": pair.frame.source_name,
        "captured_at": pair.frame.captured_at,
        "source_wait_read_ms": pair.frame.source_wait_read_ms,
        "native_shape": pair.frame.shape,
        "configuration": pair.config.to_dict(),
        "measurement": pair.result.to_dict(),
        "runtime_metrics": pair.runtime_metrics,
        "coordinate_convention": "reference to current; +x right, +y down; correction is negative displacement",
    })


def export_bundle(path: str | Path, pair: InspectionPair, diagnostic: DiagnosticSnapshot | None = None, plot_png: bytes | None = None) -> None:
    """Atomically write a self-contained .shift.zip, without Python pickle data."""
    path = Path(path)
    metadata = pair_metadata(pair)
    if diagnostic is not None:
        if diagnostic.pair.key != pair.key:
            raise ValueError("Diagnostic images and measurement must belong to the same frame pair")
        metadata["analytics"] = {"tab": diagnostic.tab, "metrics": json_safe(diagnostic.metrics)}
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("metadata.json", json.dumps(metadata, indent=2, allow_nan=False))
            for name, image in (("reference.png", pair.reference.frame.pixels), ("current.png", pair.frame.pixels)):
                success, data = cv2.imencode(".png", image)
                if not success:
                    raise ValueError("Could not encode snapshot")
                bundle.writestr(name, data.tobytes())
            if diagnostic is not None:
                data = io.BytesIO()
                np.savez_compressed(data, **diagnostic.arrays)
                bundle.writestr("analytics.npz", data.getvalue())
            if plot_png:
                bundle.writestr("analytics-view.png", plot_png)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


CSV_FIELDS = ["captured_at", "source", "frame_id", "reference_id", "config_id", "dx_px", "dy_px", "distance_px", "response", "validity", "reasons", "overlap", "residual_before", "residual_after", "processing_ms"]


def measurement_row(pair: InspectionPair) -> dict:
    result = pair.result
    return json_safe(dict(zip(CSV_FIELDS, [pair.frame.captured_at, pair.frame.source_name, pair.frame.frame_id, result.reference_id, result.config_id,
        result.dx, result.dy, result.distance, result.response, result.validity, "; ".join(result.reasons), result.overlap,
        result.residual_before, result.residual_after, result.timings.get("total_ms", 0)])))


def export_csv(path: str | Path, rows: list[dict]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_profile(path: str | Path, config: MatchConfig, tolerance: float, analytics_hz: float, camera: dict | None = None,
                 *, motion: dict | None = None, ui: dict | None = None) -> None:
    extra = {}
    if motion is not None:
        from .motion import MotionConfig
        extra["motion"] = MotionConfig.from_dict(motion).to_dict()
    if ui is not None:
        extra["ui"] = ui
    payload = json.dumps({"format_version": 1, "configuration": config.to_dict(), "tolerance_px": tolerance,
        "analytics_hz": analytics_hz, "camera": camera or {}, **extra}, indent=2, allow_nan=False)
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_profile(path: str | Path, *, include_motion: bool = False, include_ui: bool = False) -> tuple:
    """Keep the original four-item contract unless motion settings are requested."""
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if profile.get("format_version") != 1:
        raise ValueError("Unsupported settings profile version")
    config = MatchConfig.from_dict(profile["configuration"])
    tolerance, hz = float(profile.get("tolerance_px", 1)), float(profile.get("analytics_hz", 1))
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 100000 or not 0.1 <= hz <= 10:
        raise ValueError("Invalid tolerance or analytics rate")
    result = config, tolerance, hz, profile.get("camera", {})
    if include_motion:
        from .motion import MotionConfig
        motion = MotionConfig.from_dict(profile.get("motion", {}))
        result = (*result, motion.to_dict())
    if include_ui:
        ui = profile.get("ui", {})
        if not isinstance(ui, dict):
            raise ValueError("Invalid saved interface settings")
        result = (*result, ui)
    return result
