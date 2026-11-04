"""Experimental dark-slot proposals and fixed-reference identity tracking.

The detector proposes padded measurement regions, not authoritative geometry.
Tracking always compares against the selected original reference patch; ambiguous
or missing matches must be resolved by the operator rather than silently switched.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .models import Region


@dataclass(frozen=True)
class SlotCandidate:
    id: int
    bounds: tuple[int, int, int, int]
    tip: tuple[float, float]
    score: float
    contour: np.ndarray | None = None


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("Slot detection requires a nonempty grayscale or color image")
    return np.clip(image, 0, 255).astype(np.uint8)


def _field_mask(image: np.ndarray) -> np.ndarray:
    """Close inward cuts in a bright circular field without accepting its exterior."""
    h, w = image.shape
    smooth = cv2.GaussianBlur(image, (0, 0), 3)
    threshold = max(12, float(np.percentile(smooth, 95)) * 0.24)
    foreground = (smooth > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones(image.shape, np.uint8) * 255
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    area = cv2.contourArea(hull)
    # A full frame is also valid: only mask when there is an actual isolated field.
    if area < h * w * 0.12 or area > h * w * 0.96:
        return np.ones(image.shape, np.uint8) * 255
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [hull], -1, 255, -1)
    margin = max(2, round(min(h, w) * 0.008))
    return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)))


def detect_slots(image: np.ndarray, region: Region | None = None) -> list[SlotCandidate]:
    """Return ranked, padded slot candidates in original image coordinates."""
    gray = _gray(image)
    full_h, full_w = gray.shape
    x0, y0, width, height = (region or Region()).bounds(gray.shape)
    crop = gray[y0:y0 + height, x0:x0 + width]
    # Proposal geometry does not need sensor resolution. Keep expensive large
    # morphological kernels bounded, then return native pixel coordinates.
    if max(crop.shape) > 1024:
        factor = 1024 / max(crop.shape)
        small_w, small_h = max(16, round(width * factor)), max(16, round(height * factor))
        small = cv2.resize(crop, (small_w, small_h), interpolation=cv2.INTER_AREA)
        small_region = Region("circle", 0, 0, small_w, small_h) if region is not None and region.kind == "circle" else None
        sx, sy = width / small_w, height / small_h
        mapped = []
        for candidate in detect_slots(small, small_region):
            x, y, rw, rh = candidate.bounds
            left, top = x0 + math.floor(x * sx), y0 + math.floor(y * sy)
            right, bottom = min(full_w, x0 + math.ceil((x + rw) * sx)), min(full_h, y0 + math.ceil((y + rh) * sy))
            contour = None
            if candidate.contour is not None:
                contour = np.rint(candidate.contour * np.array([[[sx, sy]]]) + np.array([[[x0, y0]]])).astype(np.int32)
                contour.setflags(write=False)
            mapped.append(SlotCandidate(candidate.id, (left, top, right - left, bottom - top), (candidate.tip[0] * sx + x0, candidate.tip[1] * sy + y0), candidate.score, contour))
        return mapped
    if min(crop.shape) < 16 or float(crop.std()) < 3:
        return []
    field = _field_mask(crop)
    if region is not None and region.kind == "circle":
        circle = np.zeros_like(field)
        cv2.ellipse(circle, (width // 2, height // 2), (width // 2 - 1, height // 2 - 1), 0, 0, 360, 255, -1)
        field = cv2.bitwise_and(field, circle)
    smooth = cv2.GaussianBlur(crop, (3, 3), 0)
    size = max(15, round(min(crop.shape) * 0.18)) | 1
    baseline = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    contrast = cv2.subtract(baseline, smooth)
    inner = contrast[field > 0]
    if not inner.size:
        return []
    threshold = max(8.0, float(np.percentile(inner, 85)) * 1.35)
    dark = ((contrast >= threshold) & (field > 0)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # OpenCV represents the distance of a solid all-ones mask as infinity.
    # Explicit exterior pixels give a finite, useful image-edge distance.
    distance = cv2.distanceTransform(np.pad(field, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    proposals: list[SlotCandidate] = []
    min_width = max(3.2, min(crop.shape) * 0.008)
    min_length = max(14, min(crop.shape) * 0.045)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(30, min(crop.shape) ** 2 * 0.0002):
            continue
        rectangle = cv2.minAreaRect(contour)
        minor, major = sorted(rectangle[1])
        if minor < min_width or major < min_length or major / max(minor, 1) < 1.7:
            continue
        if minor > min(crop.shape) * 0.22 or major > max(crop.shape) * 0.95:
            continue
        points = contour[:, 0, :].astype(np.float64)
        center = points.mean(axis=0)
        _, vectors = np.linalg.eigh(np.cov((points - center).T))
        axis = vectors[:, -1]
        projection = (points - center) @ axis
        endpoints = []
        for extreme in (float(projection.min()), float(projection.max())):
            endpoint = points[np.abs(projection - extreme) <= max(1.5, minor * 0.3)].mean(axis=0)
            endpoints.append(endpoint)
        depths = [float(distance[min(height - 1, max(0, round(p[1]))), min(width - 1, max(0, round(p[0])))]) for p in endpoints]
        tip = endpoints[int(np.argmax(depths))]
        # The smoothed contrast contour extends a pixel beyond a sharp dark
        # edge. Place the marker just inside that edge, on the selected slot.
        inward = center - tip
        length = float(np.linalg.norm(inward))
        if length > 0:
            tip = tip + inward * min(0.25, max(1.5, minor * 0.15) / length)
        component_mask = np.zeros_like(dark)
        cv2.drawContours(component_mask, [contour], -1, 255, -1)
        mean_contrast = float(cv2.mean(contrast, mask=component_mask)[0])
        fill = min(1.0, area / max(major * minor, 1))
        width_weight = min(3.0, math.sqrt(minor / min_width))
        score = (mean_contrast / 255) * min(8, major / minor) * fill * width_weight * (1 + max(depths) / max(min(crop.shape), 1))
        x, y, rw, rh = cv2.boundingRect(contour)
        padding = max(10, math.ceil(minor * 1.5))
        left, top = max(0, x0 + x - padding), max(0, y0 + y - padding)
        right = min(full_w, x0 + x + rw + padding)
        bottom = min(full_h, y0 + y + rh + padding)
        native_contour = contour.copy() + np.array([[[x0, y0]]], dtype=contour.dtype)
        native_contour.setflags(write=False)
        proposals.append(SlotCandidate(0, (left, top, right - left, bottom - top), (float(tip[0] + x0), float(tip[1] + y0)), score, native_contour))
    proposals.sort(key=lambda candidate: candidate.score, reverse=True)
    return [SlotCandidate(index + 1, p.bounds, p.tip, p.score, p.contour) for index, p in enumerate(proposals)]


class SlotTracker:
    """Track the original selection with bounded search and ambiguity rejection."""

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.status = "Searching"
        self.reference_candidate: SlotCandidate | None = None
        self._template: np.ndarray | None = None
        self._last: SlotCandidate | None = None
        self._reference_shape: tuple[int, int] | None = None
        self._core_mask: np.ndarray | None = None
        self._ring_mask: np.ndarray | None = None
        self._reference_contrast = 0.0

    def select(self, candidate: SlotCandidate, reference: np.ndarray) -> None:
        gray = _gray(reference)
        x, y, width, height = candidate.bounds
        if min(width, height) < 8 or x < 0 or y < 0 or x + width > gray.shape[1] or y + height > gray.shape[0]:
            raise ValueError("Slot selection must lie inside the reference image")
        self.reference_candidate = candidate
        self._template = gray[y:y + height, x:x + width].copy()
        self._template.setflags(write=False)
        self._core_mask = None
        self._ring_mask = None
        self._reference_contrast = 0.0
        if candidate.contour is not None:
            core = np.zeros(self._template.shape, np.uint8)
            contour = candidate.contour - np.array([[[x, y]]], dtype=candidate.contour.dtype)
            cv2.drawContours(core, [contour], -1, 255, -1)
            radius = max(3, round(min(width, height) * 0.07))
            ring = cv2.subtract(cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))), core)
            if cv2.countNonZero(core) > 8 and cv2.countNonZero(ring) > 8:
                self._core_mask, self._ring_mask = core, ring
                self._reference_contrast = float(cv2.mean(self._template, mask=ring)[0] - cv2.mean(self._template, mask=core)[0])
        self._reference_shape = gray.shape
        self._last = candidate
        self.status = "Tracking"

    def update(self, image: np.ndarray) -> SlotCandidate | None:
        if self._template is None or self.reference_candidate is None or self._last is None:
            self.status = "Searching"
            return None
        gray = _gray(image)
        if gray.shape != self._reference_shape:
            self.status = "Lost"
            return None
        height, width = self._template.shape
        x, y, _, _ = self._last.bounds
        margin = max(32, round(min(gray.shape) * 0.12))
        left, top = max(0, x - margin), max(0, y - margin)
        right, bottom = min(gray.shape[1], x + width + margin), min(gray.shape[0], y + height + margin)
        search = gray[top:bottom, left:right]
        if float(search.std()) < 2 or float(self._template.std()) < 2:
            self.status = "Lost"
            return None
        correlation = cv2.matchTemplate(search, self._template, cv2.TM_CCOEFF_NORMED)
        _, best, _, location = cv2.minMaxLoc(correlation)
        if not math.isfinite(best) or best < 0.48:
            self.status = "Lost"
            return None
        px, py = location
        others = correlation.copy()
        separation = max(8, min(width, height) // 3)
        others[max(0, py - separation):py + separation + 1, max(0, px - separation):px + separation + 1] = -1
        if float(others.max()) > max(0.5, best * 0.94):
            self.status = "Ambiguous"
            return None
        if self._core_mask is not None and self._ring_mask is not None and self._reference_contrast > 8:
            matched = search[py:py + height, px:px + width]
            contrast = float(cv2.mean(matched, mask=self._ring_mask)[0] - cv2.mean(matched, mask=self._core_mask)[0])
            if contrast < max(5, self._reference_contrast * 0.35):
                self.status = "Lost"
                return None
        original = self.reference_candidate
        dx, dy = left + px - original.bounds[0], top + py - original.bounds[1]
        contour = None
        if original.contour is not None:
            contour = original.contour.copy() + np.array([[[dx, dy]]], dtype=original.contour.dtype)
            contour.setflags(write=False)
        tracked = SlotCandidate(original.id, (left + px, top + py, width, height), (original.tip[0] + dx, original.tip[1] + dy), float(best), contour)
        self._last = tracked
        self.status = "Tracking"
        return tracked
