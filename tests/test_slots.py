"""Orientation-independent proposals and fixed-reference identity tracking."""
import cv2
import numpy as np
import pytest

from measure_shift.models import Region
from measure_shift.slots import SlotCandidate, SlotTracker, detect_slots


def slot_image(angle=0, shape=(320, 320), center=(160, 155), slot_size=(100, 14)):
    image = np.full(shape, 8, np.uint8)
    cv2.circle(image, (shape[1] // 2, shape[0] // 2), round(min(shape) * 0.44), 175, -1)
    noise = np.random.default_rng(8).normal(0, 2, shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    box = cv2.boxPoints((center, slot_size, angle)).astype(np.int32)
    cv2.fillConvexPoly(image, box, 22)
    return image


@pytest.mark.parametrize("angle", [0, 30, 75, 90, 150])
def test_orientation_candidates_include_slot_and_exclude_dark_exterior(angle):
    image = slot_image(angle)
    candidates = detect_slots(image)
    assert len(candidates) == 1
    candidate = candidates[0]
    x, y, width, height = candidate.bounds
    assert x < 160 < x + width and y < 155 < y + height
    assert image[round(candidate.tip[1]), round(candidate.tip[0])] < 80
    assert np.linalg.norm(np.array(candidate.tip) - (160, 160)) < 100
    assert np.isfinite(candidate.score)


def test_competing_slots_are_exposed_and_scratches_rejected():
    image = slot_image(90, center=(110, 145), slot_size=(110, 16))
    cv2.rectangle(image, (205, 100), (221, 205), 20, -1)
    cv2.line(image, (130, 60), (240, 260), 15, 1)
    candidates = detect_slots(image)
    assert len(candidates) == 2
    assert {candidate.id for candidate in candidates} == {1, 2}


def test_roi_coordinates_remain_native():
    image = np.full((400, 500), 170, np.uint8)
    cv2.rectangle(image, (290, 170), (305, 280), 20, -1)
    candidates = detect_slots(image, Region("rectangle", 220, 130, 150, 190))
    assert len(candidates) == 1
    x, y, width, height = candidates[0].bounds
    assert x < 290 < x + width
    assert y < 170 < y + height
    assert candidates[0].tip[0] > 280


def test_reduced_detection_coordinates_and_scores_are_native_and_finite():
    image = cv2.resize(slot_image(90), (1600, 1600), interpolation=cv2.INTER_NEAREST)
    candidates = detect_slots(image)
    assert candidates
    candidate = candidates[0]
    x, y, width, height = candidate.bounds
    assert x < 800 < x + width and y < 775 < y + height
    assert candidate.tip[0] == pytest.approx(800, abs=10)
    assert np.isfinite(candidate.score)


def test_blank_and_random_noise_have_no_slot_proposals():
    assert detect_slots(np.zeros((256, 256), np.uint8)) == []
    assert detect_slots(np.full((256, 256), 150, np.uint8)) == []
    noise = np.random.default_rng(9).integers(100, 110, (256, 256), dtype=np.uint8)
    assert detect_slots(noise) == []


def test_tracking_preserves_original_identity_and_reports_loss():
    image = slot_image(30)
    candidate = detect_slots(image)[0]
    tracker = SlotTracker()
    tracker.select(candidate, image)
    original_template = tracker._template.copy()
    for dx, dy in ((12, -7), (23, -10), (31, -14)):
        moved = cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]), image.shape[::-1])
        tracked = tracker.update(moved)
        assert tracker.status == "Tracking"
        assert tracked.id == candidate.id
        assert tracked.tip == pytest.approx((candidate.tip[0] + dx, candidate.tip[1] + dy))
        assert tracker.reference_candidate is candidate
        np.testing.assert_array_equal(tracker._template, original_template)
    assert tracker.update(np.zeros_like(image)) is None
    assert tracker.status == "Lost"
    assert tracker.update(np.zeros((100, 100), np.uint8)) is None
    assert tracker.status == "Lost"
    tracker.clear()
    assert tracker.status == "Searching" and tracker.reference_candidate is None


def test_equally_good_alternatives_are_ambiguous_without_switching():
    reference = np.zeros((640, 640), np.uint8)
    patch = np.random.default_rng(3).integers(20, 220, (30, 30), dtype=np.uint8)
    reference[250:280, 250:280] = patch
    candidate = SlotCandidate(7, (250, 250, 30, 30), (265, 265), 1)
    tracker = SlotTracker()
    tracker.select(candidate, reference)
    live = reference.copy()
    live[250:280, 310:340] = patch
    assert tracker.update(live) is None
    assert tracker.status == "Ambiguous"
    assert tracker.reference_candidate is candidate


def test_disappearing_slot_is_lost_even_when_surrounding_texture_remains():
    image = slot_image(30)
    candidate = detect_slots(image)[0]
    tracker = SlotTracker()
    tracker.select(candidate, image)
    no_slot = image.copy()
    cv2.fillConvexPoly(no_slot, cv2.boxPoints(((160, 155), (104, 18), 30)).astype(np.int32), 175)
    assert tracker.update(no_slot) is None
    assert tracker.status == "Lost"
