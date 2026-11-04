"""Native-resolution display arrays without changing captured image data."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .models import InspectionPair


VIEW_MODES = ("Live", "Red / cyan overlay", "Absolute difference",
              "Compensated preview", "Compensated difference")


def render_view(pair: InspectionPair, mode: str) -> np.ndarray:
    """Render a matching pair at native size, with inverse-shift compensation.

    Overlay channels are RGB (red reference, cyan live). Compensation is a
    software preview and accepts finite uncertain estimates for inspection.
    Unsupported interpolated borders are gray in overlays and zero in errors.
    """
    if mode not in VIEW_MODES:
        raise ValueError(f"Unknown image view: {mode}")
    reference, live = pair.reference.frame.pixels, pair.frame.pixels
    if mode == "Live":
        return live.copy()
    if reference.shape != live.shape:
        raise ValueError("Reference and live image dimensions must match")
    if reference.ndim != 2 or reference.dtype != np.uint8 or live.dtype != np.uint8:
        raise ValueError("Display images must be 8-bit grayscale arrays")
    valid = None
    if mode.startswith("Compensated"):
        result = pair.result
        if not all(math.isfinite(value) for value in (result.dx, result.dy)):
            raise ValueError("No finite displacement is available for compensated preview")
        if (result.reference_id, result.frame_id, result.config_id) != pair.key:
            raise ValueError("Compensated preview requires the measurement for this frame pair")
        height, width = reference.shape
        transform = np.float32([[1, 0, -result.dx], [0, 1, -result.dy]])
        live = cv2.warpAffine(live, transform, (width, height), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
        coverage = cv2.warpAffine(np.ones(reference.shape, np.float32), transform,
                                  (width, height), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
        valid = coverage >= 1 - 1e-6
    if mode in {"Absolute difference", "Compensated difference"}:
        image = np.abs(reference.astype(np.int16) - live.astype(np.int16)).astype(np.uint8)
        if valid is not None:
            image[~valid] = 0
        return image
    image = np.dstack((reference, live, live))
    if valid is not None:
        image[~valid] = (89, 89, 89)
    return image
