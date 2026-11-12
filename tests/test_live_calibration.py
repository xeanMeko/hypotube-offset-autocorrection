"""Live calibration compares adjacent motor observations, not target offsets."""
import cv2
import numpy as np
import pytest

from measure_shift.live_calibration import LiveCalibration
from measure_shift.models import FramePacket


def texture():
    rng = np.random.default_rng(42)
    return cv2.GaussianBlur(rng.integers(15, 240, (192, 256), dtype=np.uint8), (3, 3), 0)


def translate(image, dx, dy):
    return cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]), image.shape[::-1])


@pytest.mark.parametrize("shift", [(6, -4), (-7, 3)])
def test_signed_motion_is_relative_to_pre_move_live_image(shift):
    earlier_image = texture()
    before = translate(earlier_image, 25, -12)
    after = translate(before, *shift)
    calibration = LiveCalibration()
    calibration.capture(FramePacket.from_array(before))

    result = calibration.measure(FramePacket.from_array(after))

    assert result.usable, result.reasons
    assert (result.dx, result.dy) == pytest.approx(shift, abs=.3)


def test_capture_replaces_baseline_and_invalidates_cached_observation():
    image = texture()
    calibration = LiveCalibration()
    calibration.capture(FramePacket.from_array(image))
    after = FramePacket.from_array(translate(image, 8, -4))
    first = calibration.measure(after)
    assert first.usable, first.reasons
    assert (first.dx, first.dy) == pytest.approx((8, -4), abs=.3)
    assert calibration.measure(after) is first

    calibration.capture(FramePacket.from_array(translate(image, 5, -2)))
    second = calibration.measure(after)

    assert second.usable, second.reasons
    assert (second.dx, second.dy) == pytest.approx((3, -2), abs=.3)
    assert second.reference_id != first.reference_id


def test_textureless_images_do_not_produce_usable_calibration():
    image = FramePacket.from_array(np.zeros((192, 256), dtype=np.uint8))
    calibration = LiveCalibration()
    calibration.capture(image)

    result = calibration.measure(image)

    assert not result.usable
    assert any("texture" in reason.lower() for reason in result.reasons)


def test_geometry_change_does_not_produce_usable_calibration():
    image = texture()
    calibration = LiveCalibration()
    calibration.capture(FramePacket.from_array(image))

    result = calibration.measure(FramePacket.from_array(image[:-1]))

    assert not result.usable
    assert any("geometry" in reason.lower() for reason in result.reasons)
