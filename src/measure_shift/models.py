"""Owned frame data and explicit provenance shared by workers and views."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time
from typing import Any
from uuid import uuid4

import numpy as np


def new_id() -> str:
    return uuid4().hex


@dataclass(frozen=True)
class Region:
    """ROI in native image coordinates; a circle uses its bounding square."""

    kind: str = "full"
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    def bounds(self, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        h, w = shape[:2]
        if self.kind == "full":
            return 0, 0, w, h
        if self.kind not in {"rectangle", "circle"}:
            raise ValueError("Unknown analysis region")
        x, y, rw, rh = self.x, self.y, self.width, self.height
        if min(rw, rh) < 8 or x < 0 or y < 0 or x + rw > w or y + rh > h:
            raise ValueError("Region must be inside the image and at least 8 × 8 pixels")
        return x, y, rw, rh


@dataclass(frozen=True)
class MatchConfig:
    method: str = "phase"
    region: Region = field(default_factory=Region)
    scale: float = 1.0
    min_response: float = 0.1
    min_overlap: float = 0.25
    min_texture: float = 1.0
    blur_size: int = 5
    live_region: Region | None = None
    config_id: str = field(default_factory=new_id)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MatchConfig:
        allowed = {"method", "scale", "min_response", "min_overlap", "min_texture", "blur_size"}
        kwargs = {k: v for k, v in value.items() if k in allowed}
        def region_from_dict(raw):
            region = Region(**raw)
            if region.kind not in {"full", "rectangle", "circle"}:
                raise ValueError("Unknown analysis region")
            if any(type(v) is not int for v in (region.x, region.y, region.width, region.height)):
                raise ValueError("Region coordinates must be integers")
            if region.kind != "full" and (min(region.width, region.height) < 8 or min(region.x, region.y) < 0):
                raise ValueError("Region must have nonnegative origin and dimensions of at least 8 pixels")
            return region
        kwargs["region"] = region_from_dict(value.get("region", {}))
        if value.get("live_region") is not None:
            kwargs["live_region"] = region_from_dict(value["live_region"])
        result = cls(**kwargs)
        if result.method != "phase":
            raise ValueError("Only phase correlation is supported")
        if result.scale not in (1.0, 0.5, 0.25):
            raise ValueError("Analysis scale must be 1, 1/2, or 1/4")
        if not 0 <= result.min_response <= 1 or not 0 < result.min_overlap <= 1:
            raise ValueError("Invalid quality thresholds")
        if not math.isfinite(result.min_texture) or result.min_texture < 0 or type(result.blur_size) is not int or result.blur_size < 1 or result.blur_size > 31 or result.blur_size % 2 == 0:
            raise ValueError("Invalid preprocessing settings")
        return result


@dataclass(frozen=True)
class FramePacket:
    pixels: np.ndarray
    source_name: str = "Image"
    frame_id: str = field(default_factory=new_id)
    received_at: float = field(default_factory=time.monotonic)
    captured_at: float = field(default_factory=time.time)
    source_wait_read_ms: float = 0.0
    # Acquisition-thread provenance, not a sensor exposure timestamp. Frames
    # loaded from arrays/files outside CaptureWorker have no acquisition start.
    read_started_at: float | None = None
    freshness_generation: int = 0

    @classmethod
    def from_array(cls, pixels: np.ndarray, source_name: str = "Image") -> FramePacket:
        if pixels.ndim != 2 or pixels.dtype != np.uint8:
            raise ValueError("Frames must be 8-bit grayscale images")
        owned = np.ascontiguousarray(pixels).copy()
        owned.setflags(write=False)
        return cls(owned, source_name)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.pixels.shape


@dataclass(frozen=True)
class ReferenceSnapshot:
    frame: FramePacket
    reference_id: str = field(default_factory=new_id)


@dataclass(frozen=True)
class MatchResult:
    frame_id: str
    reference_id: str
    config_id: str
    dx: float = math.nan
    dy: float = math.nan
    response: float = math.nan
    validity: str = "invalid"
    reasons: tuple[str, ...] = ()
    overlap: float = 0.0
    residual_before: float | None = None
    residual_after: float | None = None
    timings: dict[str, float] = field(default_factory=dict)
    measured_at: float = field(default_factory=time.monotonic)

    @property
    def distance(self) -> float:
        return math.hypot(self.dx, self.dy)

    @property
    def correction(self) -> tuple[float, float]:
        return -self.dx, -self.dy

    @property
    def usable(self) -> bool:
        return self.validity == "valid" and math.isfinite(self.distance)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InspectionPair:
    reference: ReferenceSnapshot
    frame: FramePacket
    config: MatchConfig
    result: MatchResult
    runtime_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str]:
        return self.reference.reference_id, self.frame.frame_id, self.config.config_id


@dataclass(frozen=True)
class DiagnosticSnapshot:
    pair: InspectionPair
    tab: str
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
