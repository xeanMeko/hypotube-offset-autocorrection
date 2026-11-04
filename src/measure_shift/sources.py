"""Owned 8-bit images from files, repeatable demos, and optional camera SDKs.

Every source is driven from one acquisition thread. SDK objects and buffers never
leave that thread: ``read`` returns a copy and returns ``None`` on a normal
timeout. Importing this module does not load either manufacturer's SDK.
"""
from __future__ import annotations

from contextlib import suppress
import ctypes
import importlib
import math
from pathlib import Path
import time
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np


class SourceError(RuntimeError):
    """An actionable input or camera error suitable for the UI."""


@runtime_checkable
class Source(Protocol):
    name: str
    finite: bool
    ended: bool

    def open(self) -> None: ...
    def read(self, timeout_ms: int = 200) -> np.ndarray | None: ...
    def close(self) -> None: ...
    def set_parameters(
        self, exposure_ms: float | None = None, gain: float | None = None,
        fps: float | None = None, pixel_clock_mhz: float | None = None,
    ) -> None: ...
    def controls(self) -> dict[str, dict[str, Any]]: ...


def read_gray(path: str | Path) -> np.ndarray:
    """Decode Unicode file paths and consistently convert to 8-bit grayscale."""
    path = Path(path).expanduser()
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise SourceError(f"Cannot read image {path}: {exc}") from exc
    if not data.size:
        raise SourceError(f"Image is empty: {path}")
    result = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if result is None:
        raise SourceError(f"Unsupported or damaged image: {path}")
    return np.ascontiguousarray(result)


def _gray_copy(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image[..., 0]
        elif image.shape[2] in (3, 4):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
        else:
            raise SourceError(f"Unsupported camera image shape: {image.shape}")
    if image.ndim != 2 or image.dtype != np.uint8:
        raise SourceError("Source did not provide an 8-bit grayscale image")
    return np.array(image, dtype=np.uint8, order="C", copy=True)


def _sdk(module: str, package: str, driver: str) -> Any:
    try:
        return importlib.import_module(module)
    except (ImportError, OSError) as exc:
        raise SourceError(
            f"{package} is unavailable. Install measure-shift[cameras] in this "
            f"environment and the manufacturer's {driver}. Details: {exc}"
        ) from exc


def _value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.split(b"\0", 1)[0].decode("utf-8", errors="replace")
    return str(value)


def _positive_fps(fps: float) -> float:
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise SourceError("Frame rate must be a finite positive number")
    return fps


class _PacedSource:
    """A timeout-bounded playback clock; no frame backlog is retained."""

    finite = False
    ended = False

    def __init__(self, fps: float = 30.0) -> None:
        self._fps = _positive_fps(fps)
        self._next_at = 0.0
        self._opened = False
        self.ended = False

    def _ready(self, timeout_ms: int) -> bool:
        if not self._opened:
            raise SourceError("Source is not open")
        wait = max(0.0, self._next_at - time.monotonic())
        budget = max(0, int(timeout_ms)) / 1000.0
        if wait > budget:
            if budget:
                time.sleep(budget)
            return False
        if wait:
            time.sleep(wait)
        self._next_at = time.monotonic() + 1.0 / self._fps
        return True

    def set_parameters(self, exposure_ms=None, gain=None, fps=None, pixel_clock_mhz=None) -> None:
        if pixel_clock_mhz is not None:
            raise SourceError("Pixel clock is only available for IDS cameras")
        if exposure_ms is not None or gain is not None:
            raise SourceError("Exposure and gain are only available for cameras")
        if fps is not None:
            value = _positive_fps(fps)
            if not 0.1 <= value <= 240:
                raise SourceError("Playback frame rate must be between 0.1 and 240 fps")
            self._fps = value
            self._next_at = 0.0

    def controls(self) -> dict[str, dict[str, Any]]:
        return {"fps": {"min": 0.1, "max": 240.0, "value": self._fps}}

    def close(self) -> None:
        self._opened = False


class DemoSource(_PacedSource):
    """Repeatable textured circular sample with a slot and known translation."""

    name = "Synthetic demo"

    def __init__(self, fps: float = 30.0, width: int = 640, height: int = 480) -> None:
        super().__init__(fps)
        self._width, self._height = width, height
        self._index = 0
        self.known_shift = (0.0, 0.0)
        rng = np.random.default_rng(419)
        y, x = np.mgrid[:height, :width]
        radius = min(width, height) * 0.42
        disk = (x - width / 2) ** 2 + (y - height / 2) ** 2 < radius ** 2
        texture = cv2.GaussianBlur(rng.normal(0, 34, (height, width)).astype(np.float32), (3, 3), 0)
        # Preserve texture at several spatial scales for realistic subpixel matching.
        texture += 9 * np.sin(x / 11) + 7 * np.cos((x + y) / 17)
        self._base = np.where(disk, np.clip(152 + texture, 0, 255), 18).astype(np.uint8)
        for _ in range(45):
            px, py = rng.integers(width // 4, 3 * width // 4), rng.integers(height // 4, 3 * height // 4)
            cv2.line(self._base, (int(px), int(py)), (int(px + 15), int(py - 6)), int(rng.integers(70, 210)), 1)
        slot_x = width // 2 - 13
        cv2.rectangle(self._base, (slot_x, height // 2), (slot_x + 26, int(height / 2 + radius)), 20, -1)
        cv2.ellipse(self._base, (width // 2, height // 2), (13, 20), 0, 180, 360, 20, -1)

    def open(self) -> None:
        self._opened = True
        self.ended = False
        self._index = 0
        self._next_at = 0.0

    def read(self, timeout_ms: int = 200) -> np.ndarray | None:
        if not self._ready(timeout_ms):
            return None
        t = self._index / self._fps
        self.known_shift = (24.0 * math.sin(t * 0.7), 16.0 * math.sin(t * 0.43))
        self._index += 1
        transform = np.array([[1, 0, self.known_shift[0]], [0, 1, self.known_shift[1]]], dtype=np.float32)
        return cv2.warpAffine(self._base, transform, (self._width, self._height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class ImageSource(_PacedSource):
    """Replay an image or directory of images; single images repeat at the clock rate."""

    _extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pgm", ".ppm"}

    def __init__(self, path: str | Path, fps: float = 30.0, loop: bool = True) -> None:
        super().__init__(fps)
        self.path = Path(path).expanduser()
        self.loop = loop
        self.finite = not loop
        self.name = f"Images: {self.path.name}"
        self._paths: list[Path] = []
        self._index = 0
        self._single: np.ndarray | None = None

    def open(self) -> None:
        self.close()
        if self.path.is_dir():
            self._paths = sorted(p for p in self.path.iterdir() if p.is_file() and p.suffix.lower() in self._extensions)
        elif self.path.is_file():
            self._paths = [self.path]
        else:
            raise SourceError(f"Image path does not exist: {self.path}")
        if not self._paths:
            raise SourceError(f"No supported images in {self.path}")
        self._single = read_gray(self._paths[0]) if len(self._paths) == 1 else None
        self._opened = True
        self.ended = False
        self._index = 0
        self._next_at = 0.0

    def read(self, timeout_ms: int = 200) -> np.ndarray | None:
        if self.ended or not self._ready(timeout_ms):
            return None
        if self._index >= len(self._paths):
            if not self.loop:
                self.ended = True
                return None
            self._index = 0
        image = self._single.copy() if self._single is not None else read_gray(self._paths[self._index])
        self._index += 1
        return image

    def close(self) -> None:
        super().close()
        self._single = None


class VideoSource(_PacedSource):
    def __init__(self, path: str | Path, loop: bool = True) -> None:
        super().__init__()
        self.path = Path(path).expanduser()
        self.name = f"Video: {self.path.name}"
        self.loop = loop
        self.finite = not loop
        self._capture: Any = None

    def open(self) -> None:
        self.close()
        if not self.path.is_file():
            raise SourceError(f"Video does not exist: {self.path}")
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self.close()
            raise SourceError(f"Cannot decode video: {self.path}")
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self._fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        self._opened = True
        self.ended = False
        self._next_at = 0.0

    def read(self, timeout_ms: int = 200) -> np.ndarray | None:
        if self.ended or not self._ready(timeout_ms):
            return None
        ok, image = self._capture.read()
        if not ok and self.loop:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = self._capture.read()
        if not ok:
            self.ended = True
            return None
        return _gray_copy(image)

    def close(self) -> None:
        super().close()
        if self._capture is not None:
            self._capture.release()
        self._capture = None


class BaslerSource:
    finite = False
    ended = False

    def __init__(self, identifier: str = "") -> None:
        self.identifier = str(identifier)
        self.name = "Basler camera"
        self._device_identity: str | None = None
        self._pylon: Any = None
        self._camera: Any = None
        self._converter: Any = None

    @property
    def device_identity(self) -> str | None:
        """The opened camera's serial, independent of default-device selection."""
        return self._device_identity

    def open(self) -> None:
        self.close()
        self._pylon = _sdk("pypylon.pylon", "pypylon", "pylon runtime")
        factory = self._pylon.TlFactory.GetInstance()
        devices = factory.EnumerateDevices()
        matches = [d for d in devices if not self.identifier or str(d.GetSerialNumber()) == self.identifier]
        if not matches:
            raise SourceError("No matching Basler camera found. Check power, cabling, and pylon Viewer discovery.")
        device = matches[0]
        try:
            self._camera = self._pylon.InstantCamera(factory.CreateDevice(device))
            self._camera.Open()
            self._device_identity = str(device.GetSerialNumber()).strip() or None
            self.name = f"Basler {device.GetModelName()} ({device.GetSerialNumber()})"
            # Conversion is explicit even for Bayer/color cameras and unpacked formats.
            self._converter = self._pylon.ImageFormatConverter()
            self._converter.OutputPixelFormat = self._pylon.PixelType_Mono8
            self._converter.OutputBitAlignment = self._pylon.OutputBitAlignment_MsbAligned
            # The measurement tool uses free-running acquisition, not a persisted trigger.
            self._set_optional("TriggerMode", "Off")
            self._set_optional("AcquisitionMode", "Continuous")
            self._camera.StartGrabbing(self._pylon.GrabStrategy_LatestImageOnly)
        except Exception as exc:
            self.close()
            raise SourceError(f"Cannot open Basler camera: {exc}") from exc

    def _set_optional(self, name: str, value: Any) -> None:
        with suppress(Exception):
            node = getattr(self._camera, name)
            if self._pylon.IsWritable(node):
                node.SetValue(value)

    def _node(self, names: tuple[str, ...], writable: bool = False) -> Any:
        if self._camera is None:
            return None
        for name in names:
            try:
                node = getattr(self._camera, name)
                if self._pylon.IsReadable(node) and (not writable or self._pylon.IsWritable(node)):
                    return node
            except Exception:
                continue
        return None

    def read(self, timeout_ms: int = 200) -> np.ndarray | None:
        if self._camera is None:
            raise SourceError("Basler camera is not open")
        grab = None
        try:
            grab = self._camera.RetrieveResult(max(0, int(timeout_ms)), self._pylon.TimeoutHandling_Return)
            if grab is None:
                return None
            if not grab.GrabSucceeded():
                raise SourceError(f"Basler acquisition failed: {grab.GetErrorDescription()}")
            converted = self._converter.Convert(grab)
            # Own the bytes before Release lets the SDK overwrite its acquisition buffer.
            return _gray_copy(converted.GetArray())
        except SourceError:
            raise
        except Exception as exc:
            raise SourceError(f"Basler read failed: {exc}") from exc
        finally:
            if grab is not None:
                grab.Release()

    def discard_pending_frames(self) -> None:
        """Restart grabbing so pre-motion SDK buffers cannot become feedback."""
        if self._camera is None:
            raise SourceError("Basler camera is not open")
        try:
            self._camera.StopGrabbing()
            self._camera.StartGrabbing(self._pylon.GrabStrategy_LatestImageOnly)
        except Exception as exc:
            raise SourceError(f"Basler fresh acquisition failed: {exc}") from exc

    def controls(self) -> dict[str, dict[str, Any]]:
        result = {}
        specs = (
            ("exposure_ms", ("ExposureTime", "ExposureTimeAbs"), 0.001),
            ("gain", ("Gain", "GainRaw"), 1.0),
            ("fps", ("AcquisitionFrameRate", "AcquisitionFrameRateAbs"), 1.0),
        )
        for key, names, scale in specs:
            try:
                node = self._node(names)
                if node is not None:
                    result[key] = {"min": float(node.GetMin()) * scale, "max": float(node.GetMax()) * scale, "value": float(node.GetValue()) * scale}
            except Exception:
                continue
        return result

    def set_parameters(self, exposure_ms=None, gain=None, fps=None, pixel_clock_mhz=None) -> None:
        if self._camera is None:
            raise SourceError("Basler camera is not open")
        errors = []
        if pixel_clock_mhz is not None:
            errors.append("pixel_clock_mhz: Pixel clock is only available for IDS cameras")
        # Set rate first, since it can change exposure bounds.
        for key, value, names, scale in (
            ("fps", fps, ("AcquisitionFrameRate", "AcquisitionFrameRateAbs"), 1.0),
            ("exposure_ms", exposure_ms, ("ExposureTime", "ExposureTimeAbs"), 1000.0),
            ("gain", gain, ("Gain", "GainRaw"), 1.0),
        ):
            if value is None:
                continue
            try:
                if not math.isfinite(float(value)):
                    raise SourceError(f"{key} must be finite")
                if key == "fps":
                    self._set_optional("AcquisitionFrameRateEnable", True)
                elif key == "exposure_ms":
                    self._set_optional("ExposureAuto", "Off")
                elif key == "gain":
                    self._set_optional("GainAuto", "Off")
                node = self._node(names, writable=True)
                if node is None:
                    raise SourceError(f"{key} is unavailable or read-only")
                converted = float(value) * scale
                if not node.GetMin() <= converted <= node.GetMax():
                    raise SourceError(f"{key} is outside the camera's supported range")
                current = node.GetValue()
                # GenICam integer gain nodes require an integer aligned to GetInc.
                if isinstance(current, (int, np.integer)):
                    step = int(node.GetInc()) if hasattr(node, "GetInc") else 1
                    converted = int(node.GetMin()) + round((converted - node.GetMin()) / max(1, step)) * max(1, step)
                node.SetValue(converted)
            except Exception as exc:
                errors.append(f"{key}: {exc}")
        if errors:
            raise SourceError("; ".join(errors))

    def close(self) -> None:
        self._device_identity = None
        camera, self._camera = self._camera, None
        self._converter = None
        if camera is not None:
            with suppress(Exception):
                if camera.IsGrabbing():
                    camera.StopGrabbing()
            with suppress(Exception):
                if camera.IsOpen():
                    camera.Close()


class IDSSource:
    """IDS uEye legacy UI camera acquisition using locked sequence buffers."""

    finite = False
    ended = False

    def __init__(self, identifier: str = "") -> None:
        self.identifier = str(identifier)
        self.name = "IDS uEye camera"
        self._device_identity: str | None = None
        self._ueye: Any = None
        self._handle: Any = None
        self._buffers: list[tuple[Any, Any]] = []
        self._queue_started = False
        self._modern_queue = False
        self._capturing = False
        self._width = self._height = self._pitch = 0
        self._bits = 8

    @property
    def device_identity(self) -> str | None:
        """The physical serial when supported; uEye device IDs are not serials."""
        return self._device_identity

    def _check(self, code: Any, operation: str) -> None:
        if _value(code) != _value(self._ueye.IS_SUCCESS):
            raise SourceError(f"IDS {operation} failed (SDK code {_value(code)})")

    def open(self) -> None:
        self.close()
        u = self._ueye = _sdk("pyueye.ueye", "pyueye", "IDS Software Suite/uEye runtime")
        identifier = int(self.identifier) if self.identifier else 0
        flags = getattr(u, "IS_USE_DEVICE_ID", 0x8000) if self.identifier else 0
        handle = u.HIDS(identifier | flags)
        self._check(u.is_InitCamera(handle, None), "initialization")
        self._handle = handle
        try:
            # Camera IDs and default-device selection may resolve to a different
            # physical unit after reconnecting. Treat serial lookup as optional:
            # older runtimes/fakes can acquire normally while the caller assigns
            # a per-session identity and conservatively invalidates calibration.
            with suppress(Exception):
                camera_info = u.CAMINFO()
                self._check(u.is_GetCameraInfo(handle, camera_info), "camera serial information")
                self._device_identity = _text(camera_info.SerNo).strip() or None
            sensor = u.SENSORINFO()
            self._check(u.is_GetSensorInfo(handle, sensor), "sensor information")
            rect = u.IS_RECT()
            self._check(u.is_AOI(handle, u.IS_AOI_IMAGE_GET_AOI, rect, u.sizeof(rect)), "image dimensions")
            self._width, self._height = int(_value(rect.s32Width)), int(_value(rect.s32Height))
            model = _text(sensor.strSensorName)
            self.name = f"IDS {model}" + (f" ({self.identifier})" if self.identifier else "")
            # The UI-5580 is a color camera. Ask the SDK for BGR8, then convert
            # after copying each pitched buffer; mono sensors use Mono8 directly.
            monochrome = _value(sensor.nColorMode) == _value(getattr(u, "IS_COLORMODE_MONOCHROME", 1))
            color_mode = u.IS_CM_MONO8 if monochrome else u.IS_CM_BGR8_PACKED
            self._bits = 8 if monochrome else 24
            self._check(u.is_SetColorMode(handle, color_mode), "8-bit color mode")
            if hasattr(u, "is_SetExternalTrigger"):
                self._check(u.is_SetExternalTrigger(handle, u.IS_SET_TRIGGER_OFF), "free-running trigger mode")
            for _ in range(3):
                memory, memory_id = u.c_mem_p(), u.int()
                self._check(u.is_AllocImageMem(handle, self._width, self._height, self._bits, memory, memory_id), "buffer allocation")
                # Record immediately so partial setup failures still free memory.
                self._buffers.append((memory, memory_id))
                self._check(u.is_AddToSequence(handle, memory, memory_id), "buffer sequence setup")
            w, h, bits, pitch = u.int(), u.int(), u.int(), u.int()
            self._check(u.is_InquireImageMem(handle, self._buffers[0][0], self._buffers[0][1], w, h, bits, pitch), "buffer pitch query")
            self._pitch = int(_value(pitch))
            if self._pitch < self._width * self._bits // 8:
                raise SourceError("IDS SDK returned an invalid image row pitch")
            self._modern_queue = all(hasattr(u, name) for name in (
                "is_ImageQueue", "IMAGEQUEUEWAITBUFFER", "IS_IMAGE_QUEUE_CMD_INIT", "IS_IMAGE_QUEUE_CMD_WAIT",
            ))
            if self._modern_queue:
                self._check(u.is_ImageQueue(handle, u.IS_IMAGE_QUEUE_CMD_INIT, None, 0), "image queue initialization")
            else:
                self._check(u.is_InitImageQueue(handle, 0), "image queue initialization")
            self._queue_started = True
            self._check(u.is_CaptureVideo(handle, u.IS_DONT_WAIT), "continuous acquisition")
            self._capturing = True
        except Exception:
            self.close()
            raise

    def _wait(self, timeout_ms: int, memory: Any, memory_id: Any) -> Any:
        u = self._ueye
        if not self._modern_queue:
            return u.is_WaitForNextImage(self._handle, max(0, int(timeout_ms)), memory, memory_id)
        # Discard old unlocked queue entries before waiting, retaining the latest.
        if all(hasattr(u, name) for name in ("IS_IMAGE_QUEUE_CMD_GET_PENDING", "IS_IMAGE_QUEUE_CMD_DISCARD_N_ITEMS")):
            pending = u.uint()
            code = u.is_ImageQueue(self._handle, u.IS_IMAGE_QUEUE_CMD_GET_PENDING, pending, u.sizeof(pending))
            if _value(code) == _value(u.IS_SUCCESS) and _value(pending) > 1:
                discard = u.uint(int(_value(pending)) - 1)
                self._check(u.is_ImageQueue(self._handle, u.IS_IMAGE_QUEUE_CMD_DISCARD_N_ITEMS, discard, u.sizeof(discard)), "discarding old frames")
        wait = u.IMAGEQUEUEWAITBUFFER()
        wait.timeout = max(0, int(timeout_ms))
        wait.ppcMem = ctypes.cast(ctypes.pointer(memory), ctypes.POINTER(u.c_char_p))
        wait.pnMemId = ctypes.pointer(memory_id)
        return u.is_ImageQueue(self._handle, u.IS_IMAGE_QUEUE_CMD_WAIT, wait, u.sizeof(wait))

    def read(self, timeout_ms: int = 200) -> np.ndarray | None:
        if self._handle is None:
            raise SourceError("IDS camera is not open")
        if not self._capturing or not self._queue_started:
            raise SourceError("IDS acquisition is stopped. Reconnect the camera to resume.")
        u = self._ueye
        memory, memory_id = u.c_mem_p(), u.int()
        code = self._wait(timeout_ms, memory, memory_id)
        if _value(code) in {_value(getattr(u, "IS_TIMED_OUT", 122)), _value(getattr(u, "IS_CAPTURE_STATUS", 3))}:
            return None
        self._check(code, "waiting for an image")
        try:
            raw = u.get_data(memory, self._width, self._height, self._bits, self._pitch, copy=True)
            rows = np.asarray(raw, dtype=np.uint8).reshape(self._height, self._pitch)
            channels = self._bits // 8
            packed = rows[:, :self._width * channels]
            image = packed.reshape(self._height, self._width) if channels == 1 else packed.reshape(self._height, self._width, channels)
            return _gray_copy(image)
        finally:
            self._check(u.is_UnlockSeqBuf(self._handle, getattr(u, "IS_IGNORE_PARAMETER", -1), memory), "buffer unlock")

    def _pixel_clock_control(self) -> dict[str, Any]:
        u, handle = self._ueye, self._handle
        current = u.uint()
        self._check(u.is_PixelClock(handle, u.IS_PIXELCLOCK_CMD_GET, current, u.sizeof(current)), "pixel clock query")
        limits = (u.uint * 3)()
        step = 1
        has_range = False
        with suppress(Exception):
            self._check(u.is_PixelClock(handle, u.IS_PIXELCLOCK_CMD_GET_RANGE, limits, u.sizeof(limits)), "pixel clock range")
            has_range = True
            step = int(_value(limits[2]))
        values = []
        with suppress(Exception):
            count = u.uint()
            self._check(u.is_PixelClock(handle, u.IS_PIXELCLOCK_CMD_GET_NUMBER, count, u.sizeof(count)), "pixel clock count")
            # The SDK documents at most 150 clocks; keep allocations bounded if
            # a broken runtime returns an invalid count.
            if 0 < int(_value(count)) <= 4096:
                clocks = (u.uint * int(_value(count)))()
                self._check(u.is_PixelClock(handle, u.IS_PIXELCLOCK_CMD_GET_LIST, clocks, u.sizeof(clocks)), "pixel clock list")
                values = sorted({int(_value(value)) for value in clocks if _value(value) > 0})
        if not values and has_range:
            low, high = int(_value(limits[0])), int(_value(limits[1]))
            # A zero increment means a discrete list is required; accepting all
            # integers between the endpoints would expose unsupported clocks.
            if low > 0 and high >= low and step > 0 and (high - low) // step < 4096:
                values = list(range(low, high + 1, step))
        if not values:
            raise SourceError("Supported pixel clocks are unavailable")
        return {"min": float(values[0]), "max": float(values[-1]),
                "value": float(_value(current)), "step": float(max(1, step)),
                "values": values}

    def controls(self) -> dict[str, dict[str, Any]]:
        if self._handle is None:
            return {}
        u, handle = self._ueye, self._handle
        result: dict[str, dict[str, Any]] = {}
        with suppress(Exception):
            result["pixel_clock_mhz"] = self._pixel_clock_control()
        with suppress(Exception):
            limits = (u.double * 3)()
            current = u.double()
            self._check(u.is_Exposure(handle, u.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE, limits, u.sizeof(limits)), "exposure range")
            self._check(u.is_Exposure(handle, u.IS_EXPOSURE_CMD_GET_EXPOSURE, current, u.sizeof(current)), "exposure query")
            result["exposure_ms"] = {"min": float(_value(limits[0])), "max": float(_value(limits[1])), "value": float(_value(current)), "step": float(_value(limits[2]))}
        with suppress(Exception):
            gain = _value(u.is_SetHardwareGain(handle, u.IS_GET_MASTER_GAIN, u.IS_IGNORE_PARAMETER, u.IS_IGNORE_PARAMETER, u.IS_IGNORE_PARAMETER))
            if 0 <= gain <= 100:
                result["gain"] = {"min": 0.0, "max": 100.0, "value": float(gain)}
        with suppress(Exception):
            low, high, inc, current = u.double(), u.double(), u.double(), u.double()
            self._check(u.is_GetFrameTimeRange(handle, low, high, inc), "frame time range")
            self._check(u.is_SetFrameRate(handle, u.IS_GET_FRAMERATE, current), "frame rate query")
            if _value(low) > 0 and _value(high) > 0:
                result["fps"] = {"min": 1.0 / float(_value(high)), "max": 1.0 / float(_value(low)), "value": float(_value(current))}
        return result

    def _reset_image_queue(self) -> None:
        """Unlock and discard images captured before a timing change."""
        u, handle = self._ueye, self._handle
        if self._modern_queue:
            self._check(u.is_ImageQueue(handle, u.IS_IMAGE_QUEUE_CMD_EXIT, None, 0), "image queue reset")
        else:
            self._check(u.is_ExitImageQueue(handle), "image queue reset")
        self._queue_started = False
        if self._modern_queue:
            self._check(u.is_ImageQueue(handle, u.IS_IMAGE_QUEUE_CMD_INIT, None, 0), "image queue restart")
        else:
            self._check(u.is_InitImageQueue(handle, 0), "image queue restart")
        self._queue_started = True

    def discard_pending_frames(self) -> None:
        """Restart both uEye queue variants after a motion-settling barrier."""
        if self._handle is None or not self._queue_started or not self._capturing:
            raise SourceError("IDS acquisition is stopped. Reconnect the camera to resume.")
        u, handle = self._ueye, self._handle
        self._check(u.is_StopLiveVideo(handle, u.IS_FORCE_VIDEO_STOP), "stopping acquisition for fresh frames")
        self._capturing = False
        self._reset_image_queue()
        self._check(u.is_CaptureVideo(handle, u.IS_DONT_WAIT), "restarting acquisition for fresh frames")
        self._capturing = True

    def set_parameters(self, exposure_ms=None, gain=None, fps=None, pixel_clock_mhz=None) -> None:
        if self._handle is None:
            raise SourceError("IDS camera is not open")
        u, handle = self._ueye, self._handle
        errors = []
        previous = self.controls() if pixel_clock_mhz is not None or fps is not None else {}
        resume_capture = False
        reset_queue = False
        clock_changed = False
        rate_changed = False
        if pixel_clock_mhz is not None:
            try:
                info = previous.get("pixel_clock_mhz")
                if info is None:
                    raise SourceError("Control is unavailable")
                value = float(pixel_clock_mhz)
                if not math.isfinite(value) or value not in info["values"]:
                    raise SourceError("Value must be one of the camera's supported pixel clocks in MHz")
                if value != info["value"]:
                    # is_PixelClock aborts an ongoing acquisition. Stop explicitly
                    # and resume only after dependent timing settings are applied.
                    if self._capturing:
                        self._check(u.is_StopLiveVideo(handle, u.IS_FORCE_VIDEO_STOP), "stopping acquisition for pixel clock")
                        self._capturing = False
                        resume_capture = True
                    reset_queue = True
                    clock = u.uint(int(value))
                    self._check(u.is_PixelClock(handle, u.IS_PIXELCLOCK_CMD_SET, clock, u.sizeof(clock)), "setting pixel clock")
                    clock_changed = True
            except Exception as exc:
                errors.append(f"pixel_clock_mhz: {exc}")
        for key, value in (("fps", fps), ("exposure_ms", exposure_ms), ("gain", gain)):
            preserve = value is None and key in previous and (
                (clock_changed and key in {"fps", "exposure_ms"})
                or (rate_changed and key == "exposure_ms")
            )
            if preserve:
                value = previous[key]["value"]
            if value is None:
                continue
            try:
                available = self.controls()
                if key not in available:
                    raise SourceError("Control is unavailable")
                info = available[key]
                if preserve:
                    value = min(info["max"], max(info["min"], value))
                if not math.isfinite(float(value)) or not info["min"] <= value <= info["max"]:
                    raise SourceError("Value is outside the camera's supported range")
                if key == "fps":
                    actual = u.double()
                    self._check(u.is_SetFrameRate(handle, u.double(float(value)), actual), "setting frame rate")
                    rate_changed = True
                elif key == "exposure_ms":
                    self._disable_auto("IS_SET_ENABLE_AUTO_SHUTTER")
                    exposure = u.double(float(value))
                    self._check(u.is_Exposure(handle, u.IS_EXPOSURE_CMD_SET_EXPOSURE, exposure, u.sizeof(exposure)), "setting exposure")
                else:
                    self._disable_auto("IS_SET_ENABLE_AUTO_GAIN")
                    self._check(u.is_SetHardwareGain(handle, int(round(value)), u.IS_IGNORE_PARAMETER, u.IS_IGNORE_PARAMETER, u.IS_IGNORE_PARAMETER), "setting gain")
            except Exception as exc:
                errors.append(f"{key}: {exc}")
        if reset_queue and self._queue_started:
            try:
                self._reset_image_queue()
            except Exception as exc:
                errors.append(f"pixel_clock_mhz: {exc}")
        if resume_capture and self._queue_started:
            try:
                self._check(u.is_CaptureVideo(handle, u.IS_DONT_WAIT), "restarting acquisition after pixel clock")
                self._capturing = True
            except Exception as exc:
                errors.append(f"pixel_clock_mhz: {exc}")
        if errors:
            raise SourceError("; ".join(errors))

    def _disable_auto(self, name: str) -> None:
        with suppress(Exception):
            zero, dummy = self._ueye.double(0), self._ueye.double(0)
            self._ueye.is_SetAutoParameter(self._handle, getattr(self._ueye, name), zero, dummy)

    def close(self) -> None:
        self._device_identity = None
        u, handle = self._ueye, self._handle
        if u is None or handle is None:
            return
        self._handle = None
        if self._capturing:
            with suppress(Exception):
                u.is_StopLiveVideo(handle, u.IS_FORCE_VIDEO_STOP)
        self._capturing = False
        if self._queue_started:
            with suppress(Exception):
                if self._modern_queue:
                    u.is_ImageQueue(handle, u.IS_IMAGE_QUEUE_CMD_EXIT, None, 0)
                else:
                    u.is_ExitImageQueue(handle)
        self._queue_started = False
        with suppress(Exception):
            u.is_ClearSequence(handle)
        for memory, memory_id in self._buffers:
            with suppress(Exception):
                u.is_FreeImageMem(handle, memory, memory_id)
        self._buffers.clear()
        with suppress(Exception):
            u.is_ExitCamera(handle)


def discover_sources(kind: str) -> list[dict[str, str]]:
    """Enumerate devices without opening cameras or changing their configuration."""
    kind = kind.strip().lower()
    if kind == "demo":
        return [{"id": "demo", "label": "Synthetic demo — known image translation"}]
    if kind in {"images", "image", "video"}:
        return []
    if kind == "basler":
        pylon = _sdk("pypylon.pylon", "pypylon", "pylon runtime")
        try:
            return [{"id": str(d.GetSerialNumber()), "label": f"{d.GetModelName()} ({d.GetSerialNumber()})"} for d in pylon.TlFactory.GetInstance().EnumerateDevices()]
        except Exception as exc:
            raise SourceError(f"Basler discovery failed: {exc}") from exc
    if kind == "ids":
        u = _sdk("pyueye.ueye", "pyueye", "IDS Software Suite/uEye runtime")
        count = u.int()
        if _value(u.is_GetNumberOfCameras(count)) != _value(u.IS_SUCCESS):
            raise SourceError("IDS camera discovery failed. Check the installed uEye runtime.")
        if not _value(count):
            return []
        camera_list = u.UEYE_CAMERA_LIST(u.UEYE_CAMERA_INFO * int(_value(count)))
        camera_list.dwCount = _value(count)
        if _value(u.is_GetCameraList(camera_list)) != _value(u.IS_SUCCESS):
            raise SourceError("IDS camera enumeration failed")
        return [{"id": str(_value(d.dwDeviceID)), "label": f"{_text(d.Model)} ({_text(d.SerNo)})"} for d in camera_list.uci[:int(_value(count))]]
    raise SourceError(f"Unknown source type: {kind}")


def create_source(kind: str, identifier: str = "", path: str = "") -> Source:
    kind = kind.strip().lower()
    if kind == "demo":
        return DemoSource()
    if kind in {"images", "image"}:
        if not path:
            raise SourceError("Select an image or image directory")
        return ImageSource(path)
    if kind == "video":
        if not path:
            raise SourceError("Select a recorded video")
        return VideoSource(path)
    if kind == "basler":
        return BaslerSource(identifier)
    if kind == "ids":
        return IDSSource(identifier)
    raise SourceError(f"Unknown source type: {kind}")
