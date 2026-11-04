from dataclasses import replace

import cv2
import numpy as np
import pytest

from measure_shift.display import render_view
from measure_shift.models import FramePacket, InspectionPair, MatchConfig, MatchResult, ReferenceSnapshot


def pair_with_shift(dx=-5, dy=3):
    reference = np.zeros((32, 48), dtype=np.uint8)
    reference[9:21, 15:27] = 200
    live = cv2.warpAffine(reference, np.float32([[1, 0, dx], [0, 1, dy]]), (48, 32))
    snapshot = ReferenceSnapshot(FramePacket.from_array(reference))
    frame = FramePacket.from_array(live)
    config = MatchConfig()
    result = MatchResult(frame.frame_id, snapshot.reference_id, config.config_id,
                         dx=dx, dy=dy, response=1, validity="valid")
    return InspectionPair(snapshot, frame, config, result)


def test_compensated_negative_displacement_and_unsupported_borders():
    pair = pair_with_shift()
    reference = pair.reference.frame.pixels.copy()
    live = pair.frame.pixels.copy()
    raw = render_view(pair, "Red / cyan overlay")
    corrected = render_view(pair, "Compensated preview")
    difference = render_view(pair, "Compensated difference")
    assert raw.shape == corrected.shape == (32, 48, 3)
    assert raw.dtype == corrected.dtype == difference.dtype == np.uint8
    assert np.any(raw[..., 0] != raw[..., 1])
    # A left/down live shift must be corrected right/up.
    np.testing.assert_array_equal(corrected[8:24, 10:35, 0], corrected[8:24, 10:35, 1])
    assert np.all(corrected[:, :5] == 89)
    assert np.all(corrected[-3:] == 89)
    assert np.count_nonzero(difference) == 0
    np.testing.assert_array_equal(pair.reference.frame.pixels, reference)
    np.testing.assert_array_equal(pair.frame.pixels, live)


def test_raw_difference_and_live_own_their_arrays():
    pair = pair_with_shift()
    difference = render_view(pair, "Absolute difference")
    assert difference.ndim == 2 and np.max(difference) == 200
    live = render_view(pair, "Live")
    live[:] = 255
    assert not np.all(pair.frame.pixels == 255)


def test_invalid_or_foreign_measurement_cannot_drive_compensation():
    pair = pair_with_shift()
    with pytest.raises(ValueError, match="finite"):
        render_view(replace(pair, result=replace(pair.result, dx=float("nan"))), "Compensated preview")
    with pytest.raises(ValueError, match="frame pair"):
        render_view(replace(pair, result=replace(pair.result, frame_id="another")), "Compensated difference")
