"""Input adapter contracts and buffer ownership, without camera hardware."""
from __future__ import annotations

import ctypes
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from measure_shift import sources


def test_unicode_image_decode_and_finite_sequence(tmp_path):
    folder = tmp_path / "Mikroskop β"
    folder.mkdir()
    first = np.arange(96, dtype=np.uint8).reshape(8, 12)
    for name, image in (("01.png", first), ("02.png", first + 17)):
        ok, data = cv2.imencode(".png", image)
        assert ok
        data.tofile(folder / name)
    np.testing.assert_array_equal(sources.read_gray(folder / "01.png"), first)
    source = sources.ImageSource(folder, loop=False)
    source.open()
    source.set_parameters(fps=240)
    np.testing.assert_array_equal(source.read(), first)
    np.testing.assert_array_equal(source.read(), first + 17)
    assert source.read() is None
    assert source.ended and source.finite
    source.close()
    source.close()


def test_single_image_returns_owned_arrays(tmp_path):
    path = tmp_path / "image.png"
    cv2.imwrite(str(path), np.full((20, 20), 37, dtype=np.uint8))
    source = sources.create_source("IMAGES", path=str(path))
    source.open()
    first = source.read()
    first[:] = 0
    assert np.all(source.read() == 37)
    source.close()
    with pytest.raises(sources.SourceError, match="not open"):
        source.read()


@pytest.mark.parametrize("kind", ["IDS", "Basler"])
def test_sdk_loading_is_deferred_and_dependency_errors_are_actionable(monkeypatch, kind):
    calls = []

    def missing(name):
        calls.append(name)
        raise ImportError("SDK missing")

    monkeypatch.setattr(sources.importlib, "import_module", missing)
    source = sources.create_source(kind)
    assert not calls
    source.close()
    with pytest.raises(sources.SourceError, match="measure-shift\\[cameras\\]"):
        source.open()
    assert len(calls) == 1


def test_demo_known_translation_and_reopen(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(sources.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(sources.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    demo = sources.create_source(" Demo ")
    demo.open()
    first = demo.read()
    assert first.shape == (480, 640) and first.dtype == np.uint8
    assert demo.known_shift == (0.0, 0.0)
    assert demo.read(timeout_ms=0) is None
    for _ in range(15):
        later = demo.read()
    measured, response = cv2.phaseCorrelate(first.astype(np.float32), later.astype(np.float32))
    assert measured == pytest.approx(demo.known_shift, abs=0.45)
    assert response > 0.5
    later[:] = 0
    demo.close()
    demo.open()
    np.testing.assert_array_equal(demo.read(), first)
    demo.close()


class FakeVideo:
    def __init__(self, frames):
        self.frames = frames
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return 120.0 if prop == cv2.CAP_PROP_FPS else 0

    def read(self):
        if self.index == len(self.frames):
            return False, None
        result = self.frames[self.index]
        self.index += 1
        return True, result

    def set(self, prop, value):
        assert prop == cv2.CAP_PROP_POS_FRAMES
        self.index = int(value)
        return True

    def release(self):
        self.released = True


def test_video_file_fps_color_copy_and_replay(tmp_path, monkeypatch):
    path = tmp_path / "recorded.avi"
    path.touch()
    color = np.zeros((12, 16, 3), dtype=np.uint8)
    color[..., 1] = 200
    capture = FakeVideo([color])
    monkeypatch.setattr(sources.cv2, "VideoCapture", lambda path: capture)
    source = sources.VideoSource(path)
    source.open()
    assert source.controls()["fps"]["value"] == 120.0
    frame = source.read()
    assert frame.shape == (12, 16)
    np.testing.assert_array_equal(frame, source.read())
    assert not source.ended
    source.close()
    assert capture.released


def test_finite_video_stops_on_eof(tmp_path, monkeypatch):
    path = tmp_path / "recorded.avi"
    path.touch()
    capture = FakeVideo([np.ones((4, 5), dtype=np.uint8)])
    monkeypatch.setattr(sources.cv2, "VideoCapture", lambda path: capture)
    source = sources.VideoSource(path, loop=False)
    source.open()
    assert source.read() is not None
    assert source.read() is None
    assert source.ended and source.finite
    source.close()


class Node:
    def __init__(self, value, low=0.0, high=100.0, readable=True, writable=True):
        self.value, self.low, self.high = value, low, high
        self.readable, self.writable = readable, writable

    def GetValue(self):
        return self.value

    def SetValue(self, value):
        self.value = value

    def GetMin(self):
        return self.low

    def GetMax(self):
        return self.high


class FakeGrab:
    def __init__(self, success=True):
        self.array = np.full((10, 12), 143, dtype=np.uint8)
        self.success = success
        self.released = False

    def GrabSucceeded(self):
        return self.success

    def GetErrorDescription(self):
        return "disconnected"

    def Release(self):
        self.released = True
        self.array[:] = 0


class FakeCamera:
    def __init__(self):
        self.opened = False
        self.grabbing = False
        self.ExposureTime = Node(2000.0, low=20, high=100000)
        self.Gain = Node(2.0, 0, 20)
        self.AcquisitionFrameRate = Node(30.0, 1, 100)
        self.AcquisitionFrameRateEnable = Node(False)
        self.ExposureAuto = Node("Off")
        self.GainAuto = Node("Off")
        self.result = FakeGrab()
        self.timeout = None

    def Open(self):
        self.opened = True

    def Close(self):
        self.opened = False

    def IsOpen(self):
        return self.opened

    def IsGrabbing(self):
        return self.grabbing

    def StartGrabbing(self, strategy):
        assert strategy == "latest"
        self.grabbing = True

    def StopGrabbing(self):
        self.grabbing = False

    def RetrieveResult(self, timeout, mode):
        assert mode == "return"
        self.timeout = timeout
        return self.result


@pytest.fixture
def fake_basler(monkeypatch):
    camera = FakeCamera()
    device = SimpleNamespace(GetSerialNumber=lambda: "123", GetModelName=lambda: "ace2")
    factory = SimpleNamespace(EnumerateDevices=lambda: [device], CreateDevice=lambda d: d)
    converter = SimpleNamespace(Convert=lambda grab: SimpleNamespace(GetArray=lambda: grab.array))
    pylon = SimpleNamespace(
        TlFactory=SimpleNamespace(GetInstance=lambda: factory),
        InstantCamera=lambda device: camera,
        ImageFormatConverter=lambda: converter,
        PixelType_Mono8="mono8", OutputBitAlignment_MsbAligned="msb",
        GrabStrategy_LatestImageOnly="latest", TimeoutHandling_Return="return",
        IsReadable=lambda node: node.readable, IsWritable=lambda node: node.writable,
    )
    monkeypatch.setattr(sources, "_sdk", lambda *args: pylon)
    return camera, converter


def test_basler_discovery_does_not_open_and_copy_precedes_release(fake_basler):
    camera, _ = fake_basler
    assert sources.discover_sources("Basler") == [{"id": "123", "label": "ace2 (123)"}]
    assert not camera.opened
    source = sources.BaslerSource("123")
    source.open()
    grab = camera.result
    frame = source.read(53)
    assert camera.timeout == 53
    assert grab.released
    assert np.all(frame == 143)
    assert not np.any(grab.array)
    source.close()
    source.close()
    assert not camera.grabbing and not camera.opened


def test_basler_default_camera_identity_uses_opened_serial_and_clears_on_close(fake_basler):
    source = sources.BaslerSource()
    assert source.device_identity is None
    source.open()
    assert source.identifier == ""
    assert source.device_identity == "123"
    device = source._pylon.TlFactory.GetInstance().EnumerateDevices()[0]
    source.close()
    assert source.device_identity is None
    device.GetSerialNumber = lambda: "456"
    source.open()
    assert source.device_identity == "456"
    source.close()


@pytest.mark.parametrize("failure", ["convert", "grab"])
def test_basler_releases_failed_frame(fake_basler, failure):
    camera, converter = fake_basler
    if failure == "grab":
        camera.result.success = False
    else:
        converter.Convert = lambda grab: (_ for _ in ()).throw(ValueError("bad format"))
    source = sources.BaslerSource()
    source.open()
    with pytest.raises(sources.SourceError):
        source.read()
    assert camera.result.released
    source.close()


def test_basler_timeout_and_independent_controls(fake_basler):
    camera, _ = fake_basler
    source = sources.BaslerSource()
    source.open()
    camera.result = None
    assert source.read(42) is None
    camera.ExposureTime.readable = False
    assert "exposure_ms" not in source.controls()
    assert "fps" in source.controls()
    with pytest.raises(sources.SourceError, match="exposure_ms"):
        source.set_parameters(exposure_ms=8, gain=5, fps=55)
    assert camera.Gain.value == 5
    assert camera.AcquisitionFrameRate.value == 55
    source.close()


def test_pixel_clock_is_rejected_by_non_ids_sources(fake_basler):
    with pytest.raises(sources.SourceError, match="only available for IDS"):
        sources.DemoSource().set_parameters(pixel_clock_mhz=20)
    camera, _ = fake_basler
    source = sources.BaslerSource()
    source.open()
    with pytest.raises(sources.SourceError, match="only available for IDS"):
        source.set_parameters(pixel_clock_mhz=20, fps=50)
    assert camera.AcquisitionFrameRate.value == 50
    source.close()


class WaitBuffer(ctypes.Structure):
    _fields_ = [("timeout", ctypes.c_uint), ("ppcMem", ctypes.POINTER(ctypes.c_char_p)), ("pnMemId", ctypes.POINTER(ctypes.c_int))]


class FakeIDS:
    IS_SUCCESS = 0
    IS_TIMED_OUT = 122
    IS_CAPTURE_STATUS = 3
    IS_USE_DEVICE_ID = 32768
    IS_COLORMODE_MONOCHROME = 1
    IS_CM_MONO8 = 6
    IS_CM_BGR8_PACKED = 7
    IS_AOI_IMAGE_GET_AOI = 1
    IS_DONT_WAIT = 0
    IS_FORCE_VIDEO_STOP = 1
    IS_IGNORE_PARAMETER = -1
    IS_SET_TRIGGER_OFF = 0
    IS_IMAGE_QUEUE_CMD_INIT = 1
    IS_IMAGE_QUEUE_CMD_WAIT = 2
    IS_IMAGE_QUEUE_CMD_EXIT = 3
    IS_IMAGE_QUEUE_CMD_GET_PENDING = 4
    IS_IMAGE_QUEUE_CMD_DISCARD_N_ITEMS = 5
    HIDS = ctypes.c_uint
    int = ctypes.c_int
    uint = ctypes.c_uint
    double = ctypes.c_double
    c_mem_p = ctypes.c_void_p
    c_char_p = ctypes.c_char_p
    sizeof = staticmethod(ctypes.sizeof)
    IMAGEQUEUEWAITBUFFER = WaitBuffer
    SENSORINFO = staticmethod(lambda: SimpleNamespace(strSensorName=b"UI-5580", nColorMode=2))

    class IS_RECT(ctypes.Structure):
        _fields_ = [("s32Width", ctypes.c_int), ("s32Height", ctypes.c_int)]

    def __init__(self):
        self.calls = []
        self.buffers = []
        self.bits = 24
        self.wait_code = self.IS_SUCCESS
        self.pending = 3
        self.bad_data = False
        self.fail_add = False

    def is_InitCamera(self, handle, _):
        self.calls.append(("init", handle.value))
        return 0

    def is_GetSensorInfo(self, handle, sensor):
        return 0

    def is_AOI(self, handle, cmd, rect, size):
        rect.s32Width, rect.s32Height = 4, 3
        return 0

    def is_SetColorMode(self, handle, mode):
        self.calls.append(("color", mode))
        return 0

    def is_SetExternalTrigger(self, handle, mode):
        return 0

    def is_AllocImageMem(self, handle, width, height, bits, memory, memory_id):
        self.bits = bits
        buffer = ctypes.create_string_buffer((width * bits // 8 + 2) * height)
        memory.value = ctypes.addressof(buffer)
        memory_id.value = len(self.buffers) + 1
        self.buffers.append(buffer)
        return 0

    def is_AddToSequence(self, handle, memory, memory_id):
        return 4 if self.fail_add else 0

    def is_InquireImageMem(self, handle, memory, memory_id, width, height, bits, pitch):
        width.value, height.value, bits.value, pitch.value = 4, 3, self.bits, 4 * self.bits // 8 + 2
        return 0

    def is_ImageQueue(self, handle, command, param, size):
        self.calls.append(("queue", command))
        if command == self.IS_IMAGE_QUEUE_CMD_GET_PENDING:
            param.value = self.pending
        elif command == self.IS_IMAGE_QUEUE_CMD_DISCARD_N_ITEMS:
            self.calls.append(("discard", param.value))
        elif command == self.IS_IMAGE_QUEUE_CMD_WAIT:
            if self.wait_code:
                return self.wait_code
            ctypes.cast(param.ppcMem, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.addressof(self.buffers[0])
            param.pnMemId[0] = 1
            self.calls.append(("wait_ms", param.timeout))
        return 0

    def is_CaptureVideo(self, handle, mode):
        return 0

    def get_data(self, memory, width, height, bits, pitch, copy):
        if self.bad_data:
            raise ValueError("decode failure")
        assert copy
        rows = np.ctypeslib.as_array(ctypes.cast(memory, ctypes.POINTER(ctypes.c_uint8)), (height * pitch,)).reshape(height, pitch)
        rows[:, :width * bits // 8] = 100
        rows[:, width * bits // 8:] = 255
        return rows.flatten()

    def is_UnlockSeqBuf(self, handle, ignored, memory):
        self.calls.append(("unlock", ignored))
        ctypes.memset(memory, 0, len(self.buffers[0]))
        return 0

    def is_StopLiveVideo(self, handle, mode):
        self.calls.append(("stop",))
        return 0

    def is_ClearSequence(self, handle):
        self.calls.append(("clear",))
        return 0

    def is_FreeImageMem(self, handle, memory, memory_id):
        self.calls.append(("free", memory_id.value))
        return 0

    def is_ExitCamera(self, handle):
        self.calls.append(("exit",))
        return 0


@pytest.fixture
def fake_ids(monkeypatch):
    sdk = FakeIDS()
    monkeypatch.setattr(sources, "_sdk", lambda *args: sdk)
    return sdk


def test_ids_pitch_color_copy_drop_stale_frames_and_cleanup(fake_ids):
    source = sources.IDSSource("42")
    source.open()
    assert ("init", 42 | fake_ids.IS_USE_DEVICE_ID) in fake_ids.calls
    assert len(fake_ids.buffers) == 3
    frame = source.read(61)
    assert frame.shape == (3, 4)
    assert frame.dtype == np.uint8
    assert np.all(frame == 100)  # Padding bytes (255) must never become image pixels.
    assert ("wait_ms", 61) in fake_ids.calls
    assert ("discard", 2) in fake_ids.calls
    assert ("unlock", -1) in fake_ids.calls
    source.close()
    source.close()
    assert fake_ids.calls.count(("exit",)) == 1
    assert len([call for call in fake_ids.calls if call[0] == "free"]) == 3
    assert fake_ids.calls.index(("clear",)) < fake_ids.calls.index(("free", 1))


@pytest.mark.parametrize("identifier", ["", "42"])
def test_ids_identity_uses_physical_serial_instead_of_selection_id(fake_ids, identifier):
    fake_ids.CAMINFO = lambda: SimpleNamespace(SerNo=b"")

    def camera_info(handle, info):
        info.SerNo = b"4000123456\0"
        return fake_ids.IS_SUCCESS

    fake_ids.is_GetCameraInfo = camera_info
    source = sources.IDSSource(identifier)
    assert source.device_identity is None
    source.open()
    assert source.device_identity == "4000123456"
    source.close()
    assert source.device_identity is None


@pytest.mark.parametrize("serial,code", [(b"", 0), (b"4000123456", 1)])
def test_ids_unavailable_or_empty_serial_keeps_identity_unresolved(fake_ids, serial, code):
    fake_ids.CAMINFO = lambda: SimpleNamespace(SerNo=serial)
    fake_ids.is_GetCameraInfo = lambda *_: code
    source = sources.IDSSource("42")
    source.open()
    assert source.device_identity is None
    assert source.read() is not None
    source.close()


def test_ids_missing_optional_serial_api_does_not_prevent_acquisition(fake_ids):
    source = sources.IDSSource()
    source.open()
    assert source.device_identity is None
    assert source.read() is not None
    source.close()


def test_ids_timeout_does_not_unlock_unowned_buffer(fake_ids):
    source = sources.IDSSource()
    source.open()
    fake_ids.wait_code = fake_ids.IS_TIMED_OUT
    assert source.read(5) is None
    assert not any(call[0] == "unlock" for call in fake_ids.calls)
    source.close()


def test_ids_decode_failure_unlocks_buffer(fake_ids):
    source = sources.IDSSource()
    source.open()
    fake_ids.bad_data = True
    with pytest.raises(ValueError, match="decode failure"):
        source.read()
    assert ("unlock", -1) in fake_ids.calls
    source.close()


def test_ids_partial_setup_failure_releases_all_allocated_buffers(fake_ids):
    fake_ids.fail_add = True
    source = sources.IDSSource()
    with pytest.raises(sources.SourceError, match="sequence setup"):
        source.open()
    assert ("free", 1) in fake_ids.calls
    assert ("exit",) in fake_ids.calls
    assert not source._buffers


def test_ids_legacy_queue_fallback(fake_ids, monkeypatch):
    monkeypatch.delattr(FakeIDS, "IMAGEQUEUEWAITBUFFER")
    fake_ids.is_InitImageQueue = lambda handle, ignored: 0
    fake_ids.is_ExitImageQueue = lambda handle: fake_ids.calls.append(("legacy_exit",)) or 0

    def wait(handle, timeout, memory, memory_id):
        memory.value = ctypes.addressof(fake_ids.buffers[0])
        memory_id.value = 1
        return 0

    fake_ids.is_WaitForNextImage = wait
    source = sources.IDSSource()
    source.open()
    assert not source._modern_queue
    assert np.all(source.read() == 100)
    source.close()
    assert ("legacy_exit",) in fake_ids.calls


def test_ids_controls_work_independently(fake_ids):
    fake_ids.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE = 10
    fake_ids.IS_EXPOSURE_CMD_GET_EXPOSURE = 11
    fake_ids.IS_GET_MASTER_GAIN = 1000
    fake_ids.IS_GET_FRAMERATE = -1
    fake_ids.is_Exposure = lambda *args: 125  # Unsupported exposure must not hide FPS/gain.
    values = {"gain": 7, "fps": 30.0}

    def gain(handle, value, *_):
        if value == fake_ids.IS_GET_MASTER_GAIN:
            return values["gain"]
        values["gain"] = value
        return 0

    def rate(handle, value, current):
        if value != fake_ids.IS_GET_FRAMERATE:
            values["fps"] = value.value
        current.value = values["fps"]
        return 0

    def limits(handle, low, high, increment):
        low.value, high.value, increment.value = 0.01, 1, 0.001
        return 0

    fake_ids.is_SetHardwareGain = gain
    fake_ids.is_SetFrameRate = rate
    fake_ids.is_GetFrameTimeRange = limits
    source = sources.IDSSource()
    source.open()
    assert set(source.controls()) == {"gain", "fps"}
    with pytest.raises(sources.SourceError, match="exposure_ms"):
        source.set_parameters(fps=50.0, exposure_ms=2.0, gain=18.0)
    assert values == {"gain": 18, "fps": 50.0}
    source.close()


@pytest.fixture
def ids_timing(fake_ids):
    """A camera whose timing bounds change with pixel clock and frame rate."""
    sdk = fake_ids
    sdk.IS_PIXELCLOCK_CMD_GET = 20
    sdk.IS_PIXELCLOCK_CMD_GET_RANGE = 21
    sdk.IS_PIXELCLOCK_CMD_GET_NUMBER = 22
    sdk.IS_PIXELCLOCK_CMD_GET_LIST = 23
    sdk.IS_PIXELCLOCK_CMD_SET = 24
    sdk.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE = 30
    sdk.IS_EXPOSURE_CMD_GET_EXPOSURE = 31
    sdk.IS_EXPOSURE_CMD_SET_EXPOSURE = 32
    sdk.IS_GET_FRAMERATE = -1
    sdk.IS_GET_MASTER_GAIN = 1000
    state = SimpleNamespace(clock=20, fps=20.0, exposure=10.0, gain=3,
                            values=[10, 20, 40], clock_range=(10, 40, 0),
                            list_available=True, range_available=True,
                            set_code=0, stop_code=0, capture_code=0)

    def pixel_clock(handle, command, param, size):
        if command == sdk.IS_PIXELCLOCK_CMD_GET:
            param.value = state.clock
        elif command == sdk.IS_PIXELCLOCK_CMD_GET_RANGE:
            if not state.range_available:
                return 125
            param[:] = state.clock_range
        elif command == sdk.IS_PIXELCLOCK_CMD_GET_NUMBER:
            if not state.list_available:
                return 125
            param.value = len(state.values)
        elif command == sdk.IS_PIXELCLOCK_CMD_GET_LIST:
            assert size == len(state.values) * ctypes.sizeof(ctypes.c_uint)
            param[:] = state.values
        elif command == sdk.IS_PIXELCLOCK_CMD_SET:
            sdk.calls.append(("clock", param.value))
            if state.set_code:
                return state.set_code
            state.clock = param.value
        return 0

    def frame_limits(handle, low, high, increment):
        sdk.calls.append(("rate_range", state.clock))
        low.value, high.value, increment.value = 1.0 / (2 * state.clock), 1.0, 0.001
        return 0

    def frame_rate(handle, value, current):
        if value != sdk.IS_GET_FRAMERATE:
            sdk.calls.append(("rate", value.value))
            assert 1 <= value.value <= state.clock * 2
            state.fps = value.value
        current.value = state.fps
        return 0

    def exposure(handle, command, param, size):
        if command == sdk.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE:
            sdk.calls.append(("exposure_range", state.clock, state.fps))
            param[:] = (0.01, min(float(state.clock), 1000.0 / state.fps), 0.01)
        elif command == sdk.IS_EXPOSURE_CMD_GET_EXPOSURE:
            param.value = state.exposure
        elif command == sdk.IS_EXPOSURE_CMD_SET_EXPOSURE:
            sdk.calls.append(("exposure", param.value))
            assert 0.01 <= param.value <= min(state.clock, 1000.0 / state.fps)
            state.exposure = param.value
        return 0

    def gain(handle, value, *_):
        if value == sdk.IS_GET_MASTER_GAIN:
            return state.gain
        sdk.calls.append(("gain", value))
        state.gain = value
        return 0

    queue = sdk.is_ImageQueue

    def image_queue(handle, command, param, size):
        result = queue(handle, command, param, size)
        if command == sdk.IS_IMAGE_QUEUE_CMD_EXIT:
            state.stale_cleared = True
            sdk.pending = 0
        return result

    def capture(handle, mode):
        sdk.calls.append(("capture",))
        return state.capture_code

    def stop(handle, mode):
        sdk.calls.append(("stop",))
        return state.stop_code

    sdk.is_PixelClock = pixel_clock
    sdk.is_GetFrameTimeRange = frame_limits
    sdk.is_SetFrameRate = frame_rate
    sdk.is_Exposure = exposure
    sdk.is_SetHardwareGain = gain
    sdk.is_ImageQueue = image_queue
    sdk.is_CaptureVideo = capture
    sdk.is_StopLiveVideo = stop
    return sdk, state


def test_ids_pixel_clock_discrete_values_and_exposure_step(ids_timing):
    sdk, _ = ids_timing
    source = sources.IDSSource()
    source.open()
    info = source.controls()
    assert info["pixel_clock_mhz"] == {
        "min": 10.0, "max": 40.0, "value": 20.0, "step": 1.0,
        "values": [10, 20, 40],
    }
    assert info["exposure_ms"]["step"] == 0.01
    source.close()


@pytest.mark.parametrize("range_available", [False, True])
def test_ids_pixel_clock_list_does_not_require_a_supported_range(ids_timing, range_available):
    _, state = ids_timing
    state.range_available = range_available
    state.values = [40, 10, 20, 20]
    source = sources.IDSSource()
    source.open()
    assert source.controls()["pixel_clock_mhz"]["values"] == [10, 20, 40]
    source.close()


def test_ids_pixel_clock_range_fallback_and_unavailable_discrete_list(ids_timing):
    _, state = ids_timing
    state.list_available = False
    state.clock_range = (10, 40, 10)
    source = sources.IDSSource()
    source.open()
    assert source.controls()["pixel_clock_mhz"]["values"] == [10, 20, 30, 40]
    assert source.controls()["pixel_clock_mhz"]["step"] == 10
    state.clock_range = (10, 40, 0)
    assert "pixel_clock_mhz" not in source.controls()
    assert "fps" in source.controls() and "exposure_ms" in source.controls()
    source.close()


@pytest.mark.parametrize("clock", [15, 20.1, -1, 0, float("nan"), float("inf")])
def test_ids_rejects_unsupported_pixel_clocks_without_stopping_capture(ids_timing, clock):
    sdk, state = ids_timing
    source = sources.IDSSource()
    source.open()
    sdk.calls.clear()
    with pytest.raises(sources.SourceError, match="pixel_clock_mhz"):
        source.set_parameters(pixel_clock_mhz=clock)
    assert state.clock == 20
    assert not any(call[0] in {"stop", "clock", "capture"} for call in sdk.calls)
    assert source._capturing
    source.close()


def test_ids_sets_clock_before_refreshing_dependent_timing_and_restarts_with_empty_queue(ids_timing):
    sdk, state = ids_timing
    source = sources.IDSSource()
    source.open()
    sdk.calls.clear()
    source.set_parameters(pixel_clock_mhz=40, fps=60, exposure_ms=12, gain=9)
    assert (state.clock, state.fps, state.exposure, state.gain) == (40, 60, 12, 9)
    calls = sdk.calls
    assert calls.index(("stop",)) < calls.index(("clock", 40))
    assert calls.index(("clock", 40)) < calls.index(("rate_range", 40)) < calls.index(("rate", 60))
    assert calls.index(("rate", 60)) < calls.index(("exposure_range", 40, 60)) < calls.index(("exposure", 12))
    assert calls.index(("exposure", 12)) < calls.index(("gain", 9))
    assert calls.index(("queue", sdk.IS_IMAGE_QUEUE_CMD_EXIT)) < calls.index(("queue", sdk.IS_IMAGE_QUEUE_CMD_INIT)) < calls.index(("capture",))
    assert state.stale_cleared and sdk.pending == 0
    assert source._capturing
    assert source.read() is not None
    source.close()


@pytest.mark.parametrize("fps, exposure, expected_fps, expected_exposure", [(70, 12, 40, 12), (20, 30, 20, 20)])
def test_ids_clock_only_change_preserves_timing_within_new_ranges(ids_timing, fps, exposure, expected_fps, expected_exposure):
    _, state = ids_timing
    state.clock, state.fps, state.exposure = 40, fps, exposure
    source = sources.IDSSource()
    source.open()
    source.set_parameters(pixel_clock_mhz=20)
    assert state.fps == expected_fps and state.exposure == expected_exposure
    assert source._capturing
    source.close()


def test_ids_fps_only_change_restores_exposure_after_sdk_resets_it(ids_timing):
    sdk, state = ids_timing
    state.clock, state.fps, state.exposure = 40, 20, 30
    frame_rate = sdk.is_SetFrameRate

    def reset_exposure_on_rate_change(handle, value, current):
        result = frame_rate(handle, value, current)
        if value != sdk.IS_GET_FRAMERATE:
            state.exposure = 0.01
        return result

    sdk.is_SetFrameRate = reset_exposure_on_rate_change
    source = sources.IDSSource()
    source.open()
    sdk.calls.clear()
    source.set_parameters(fps=60)
    assert state.fps == 60
    assert state.exposure == pytest.approx(1000 / 60)
    assert sdk.calls.index(("rate", 60)) < sdk.calls.index(("exposure", state.exposure))
    assert not any(call[0] in {"stop", "capture", "clock"} for call in sdk.calls)
    source.close()


def test_ids_same_clock_does_not_interrupt_acquisition(ids_timing):
    sdk, _ = ids_timing
    source = sources.IDSSource()
    source.open()
    sdk.calls.clear()
    source.set_parameters(pixel_clock_mhz=20)
    assert not any(call[0] in {"stop", "clock", "capture", "rate", "exposure"} for call in sdk.calls)
    source.close()


@pytest.mark.parametrize("failure", ["clock", "exposure", "rate", "restart", "stop"])
def test_ids_pixel_clock_errors_keep_acquisition_state_accurate(ids_timing, failure):
    sdk, state = ids_timing
    source = sources.IDSSource()
    source.open()
    sdk.calls.clear()
    requested = {"pixel_clock_mhz": 40}
    if failure == "clock":
        state.set_code = 125
    elif failure == "exposure":
        requested["exposure_ms"] = 1000
    elif failure == "rate":
        requested["fps"] = 1000
    elif failure == "restart":
        state.capture_code = 125
    else:
        state.stop_code = 125
    with pytest.raises(sources.SourceError):
        source.set_parameters(**requested)
    assert source._capturing is (failure != "restart")
    if failure == "stop":
        assert state.clock == 20
        assert not any(call[0] == "clock" for call in sdk.calls)
    else:
        assert ("capture",) in sdk.calls
        assert state.stale_cleared
    if failure == "restart":
        with pytest.raises(sources.SourceError, match="acquisition is stopped.*Reconnect"):
            source.read()
    source.close()


def test_ids_queue_restart_failure_requires_reconnect_instead_of_waiting_forever(ids_timing):
    sdk, _ = ids_timing
    source = sources.IDSSource()
    source.open()
    sdk.calls.clear()
    queue = sdk.is_ImageQueue

    def fail_queue_restart(handle, command, param, size):
        if command == sdk.IS_IMAGE_QUEUE_CMD_INIT:
            return 125
        return queue(handle, command, param, size)

    sdk.is_ImageQueue = fail_queue_restart
    with pytest.raises(sources.SourceError, match="image queue restart"):
        source.set_parameters(pixel_clock_mhz=40)
    assert not source._capturing and not source._queue_started
    assert ("capture",) not in sdk.calls
    with pytest.raises(sources.SourceError, match="acquisition is stopped.*Reconnect"):
        source.read()
    source.close()


def test_ids_clock_change_resets_legacy_queue(ids_timing, monkeypatch):
    sdk, state = ids_timing
    monkeypatch.delattr(FakeIDS, "IMAGEQUEUEWAITBUFFER")
    sdk.is_InitImageQueue = lambda handle, ignored: sdk.calls.append(("legacy_init",)) or 0
    sdk.is_ExitImageQueue = lambda handle: sdk.calls.append(("legacy_exit",)) or 0
    source = sources.IDSSource()
    source.open()
    sdk.calls.clear()
    source.set_parameters(pixel_clock_mhz=40)
    assert state.clock == 40 and source._capturing
    assert sdk.calls.index(("legacy_exit",)) < sdk.calls.index(("legacy_init",)) < sdk.calls.index(("capture",))
    source.close()


def test_ids_discovery_uses_struct_array_without_opening(monkeypatch):
    class Info(ctypes.Structure):
        _fields_ = [("dwDeviceID", ctypes.c_uint), ("Model", ctypes.c_char * 32), ("SerNo", ctypes.c_char * 16)]

    def camera_list(array_type):
        assert issubclass(array_type, ctypes.Array)
        return SimpleNamespace(dwCount=0, uci=array_type())

    def get_cameras(camera_list):
        camera_list.uci[0].dwDeviceID = 42
        camera_list.uci[0].Model = b"UI-5580RE"
        camera_list.uci[0].SerNo = b"410"
        return 0

    sdk = SimpleNamespace(
        int=ctypes.c_int, IS_SUCCESS=0, UEYE_CAMERA_INFO=Info,
        UEYE_CAMERA_LIST=camera_list,
        is_GetNumberOfCameras=lambda count: setattr(count, "value", 1) or 0,
        is_GetCameraList=get_cameras,
    )
    monkeypatch.setattr(sources, "_sdk", lambda *args: sdk)
    assert sources.discover_sources("ids") == [{"id": "42", "label": "UI-5580RE (410)"}]


def test_input_validation(tmp_path):
    with pytest.raises(sources.SourceError, match="Unknown"):
        sources.create_source("not a source")
    with pytest.raises(sources.SourceError, match="Select"):
        sources.create_source("Video")
    with pytest.raises(sources.SourceError, match="No supported images"):
        sources.ImageSource(tmp_path).open()
    corrupt = tmp_path / "invalid.png"
    corrupt.write_bytes(b"not an image")
    with pytest.raises(sources.SourceError, match="Unsupported or damaged"):
        sources.read_gray(corrupt)
    with pytest.raises(sources.SourceError, match="finite positive"):
        sources.DemoSource(fps=float("nan"))
