from dataclasses import replace
import csv
import io
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import pytest

from measure_shift.models import DiagnosticSnapshot, FramePacket, InspectionPair, MatchConfig, MatchResult, ReferenceSnapshot, Region
from measure_shift.storage import CSV_FIELDS, export_bundle, export_csv, load_profile, measurement_row, save_profile


def test_failed_profile_replace_preserves_previous_settings_and_removes_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    save_profile(path, MatchConfig(), 2, 3)
    before = path.read_bytes()

    def failed_replace(self, target):
        raise OSError("Cannot replace settings")

    monkeypatch.setattr(Path, "replace", failed_replace)
    with pytest.raises(OSError, match="Cannot replace settings"):
        save_profile(path, MatchConfig(scale=.5), 5, 6)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def make_pair():
    image = np.arange(24 * 32, dtype=np.uint8).reshape(24, 32)
    reference = ReferenceSnapshot(FramePacket.from_array(image, "reference"))
    frame = FramePacket.from_array(image, 'camera, "test"')
    config = MatchConfig()
    result = MatchResult(frame.frame_id, reference.reference_id, config.config_id)
    return InspectionPair(reference, frame, config, result)


def test_bundle_preserves_pair_pixels_arrays_and_strict_json(tmp_path):
    pair = make_pair()
    diagnostic = DiagnosticSnapshot(pair, "Peak", arrays={"surface": np.array([[0.25, -0.5]], np.float32)},
                                     metrics={"not_finite": np.float64(np.nan), "count": np.int64(2)})
    destination = tmp_path / "alignment.shift.zip"
    export_bundle(destination, pair, diagnostic, b"example plot")
    with zipfile.ZipFile(destination) as archive:
        text = archive.read("metadata.json").decode()
        metadata = json.loads(text, parse_constant=lambda value: pytest.fail(f"Nonstandard JSON constant: {value}"))
        assert metadata["reference_id"] == pair.reference.reference_id
        assert metadata["frame_id"] == pair.frame.frame_id
        assert metadata["measurement"]["dx"] is None
        assert metadata["measurement"]["response"] is None
        assert metadata["analytics"]["metrics"]["not_finite"] is None
        assert metadata["analytics"]["metrics"]["count"] == 2
        for member, expected in [("reference.png", pair.reference.frame.pixels), ("current.png", pair.frame.pixels)]:
            decoded = cv2.imdecode(np.frombuffer(archive.read(member), np.uint8), cv2.IMREAD_GRAYSCALE)
            np.testing.assert_array_equal(decoded, expected)
        with np.load(io.BytesIO(archive.read("analytics.npz")), allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["surface"], diagnostic.arrays["surface"])
        assert archive.read("analytics-view.png") == b"example plot"
    assert list(tmp_path.glob("*.tmp")) == []


def test_bundle_rejects_diagnostics_from_another_pair_before_creating_file(tmp_path):
    pair = make_pair()
    diagnostic = DiagnosticSnapshot(make_pair(), "Inputs")
    destination = tmp_path / "wrong.shift.zip"
    with pytest.raises(ValueError, match="same frame pair"):
        export_bundle(destination, pair, diagnostic)
    assert list(tmp_path.iterdir()) == []


def test_bundle_rejects_measurement_with_foreign_provenance(tmp_path):
    pair = make_pair()
    inconsistent = replace(pair, result=replace(pair.result, frame_id="other frame"))
    with pytest.raises(ValueError, match="[Pp]rovenance|frame pair|measurement"):
        export_bundle(tmp_path / "wrong.shift.zip", inconsistent)


def test_csv_preserves_fields_quotes_and_blanks_invalid_numbers(tmp_path):
    pair = make_pair()
    invalid_row = measurement_row(pair)
    assert invalid_row["dx_px"] is None
    valid = replace(pair, result=replace(pair.result, dx=-3, dy=4, response=0.8,
                                        validity="uncertain", reasons=("weak", "check visually"),
                                        timings={"total_ms": 12.5}))
    destination = tmp_path / "measurements.csv"
    export_csv(destination, [invalid_row, measurement_row(valid)])
    with destination.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == CSV_FIELDS
        rows = list(reader)
    assert rows[0]["source"] == 'camera, "test"'
    assert rows[0]["dx_px"] == rows[0]["response"] == ""
    assert rows[1]["dx_px"] == "-3"
    assert rows[1]["distance_px"] == "5.0"
    assert rows[1]["reasons"] == "weak; check visually"
    assert rows[1]["processing_ms"] == "12.5"


def test_profile_roundtrip(tmp_path):
    config = MatchConfig(region=Region("rectangle", 5, 8, 48, 40), scale=0.5)
    path = tmp_path / "settings.json"
    settings = {"pixel_clock_mhz": 24, "fps": 3.0, "exposure_ms": 200.0}
    save_profile(path, config, tolerance=1.5, analytics_hz=0.5, camera=settings)
    restored, tolerance, hz, camera = load_profile(path)
    assert replace(restored, config_id=config.config_id) == config
    assert (tolerance, hz, camera) == (1.5, 0.5, settings)


def test_motion_profile_roundtrip_keeps_original_loader_contract(tmp_path):
    from measure_shift.motion import MotionConfig
    motion = MotionConfig(motor_current_ma=800, jog_distance=2, jog_speed=20,
                          jog_unit="pixels", repeat_hz=3.5,
                          pixels_per_step=-.05, calibration_shape=(480, 640),
                          calibration_source_id="ids:serial:example")
    path = tmp_path / "motion.json"
    save_profile(path, MatchConfig(), 1, 1, motion=motion.to_dict())
    assert len(load_profile(path)) == 4
    assert MotionConfig.from_dict(load_profile(path, include_motion=True)[4]) == motion
    legacy = tmp_path / "legacy.json"
    save_profile(legacy, MatchConfig(), 1, 1)
    assert MotionConfig.from_dict(load_profile(legacy, include_motion=True)[4]) == MotionConfig()


@pytest.mark.parametrize("region", [
    {"kind": "rectangle", "x": 0, "y": 0, "width": -1, "height": 20},
    {"kind": "rectangle", "x": -3, "y": 0, "width": 20, "height": 20},
    {"kind": "rectangle", "x": 0, "y": 0, "width": 20.5, "height": 20},
    {"kind": "unknown", "x": 0, "y": 0, "width": 20, "height": 20},
])
def test_profile_rejects_invalid_region_geometry(tmp_path, region):
    path = tmp_path / "bad.json"
    value = {"format_version": 1, "configuration": {"region": region}}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_profile(path)


@pytest.mark.parametrize("field,value", [("tolerance_px", float("nan")), ("analytics_hz", 0), ("format_version", 2)])
def test_profile_rejects_invalid_limits_and_versions(tmp_path, field, value):
    path = tmp_path / "bad.json"
    profile = {"format_version": 1, "configuration": {}}
    profile[field] = value
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError):
        load_profile(path)
