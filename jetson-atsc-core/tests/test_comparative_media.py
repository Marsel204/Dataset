"""Unit and integration tests for comparative media deliverables."""
from pathlib import Path
import cv2
import pytest

MEDIA_DIR = Path(__file__).resolve().parent.parent / "data" / "comparative_media"

SCENARIOS = [
    ("compare_stress_night_arterial.mp4", "snapshot_stress_night_arterial.jpg"),
    ("compare_stress_rain_wet_road.mp4", "snapshot_stress_rain_wet_road.jpg"),
    ("compare_stress_congestion_gridlock.mp4", "snapshot_stress_congestion_gridlock.jpg"),
    ("compare_stress_indonesia_bandung_pasteur.mp4", "snapshot_stress_indonesia_bandung_pasteur.jpg"),
    ("compare_stress_indonesia_bekasi.mp4", "snapshot_stress_indonesia_bekasi.jpg"),
]


@pytest.mark.parametrize("video_name,snapshot_name", SCENARIOS)
def test_comparative_video_exists_and_decodable(video_name: str, snapshot_name: str) -> None:
    """Verify each comparative MP4 video file exists, is non-empty, and decodes."""
    video_path = MEDIA_DIR / video_name
    assert video_path.exists(), f"Missing comparative video: {video_path}"
    assert video_path.stat().st_size > 10_000, f"Comparative video is suspiciously small: {video_path}"

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"OpenCV failed to open: {video_path}"

    ret, frame = cap.read()
    cap.release()

    assert ret is True, f"Failed to decode first frame from: {video_path}"
    assert frame is not None
    assert frame.shape[0] == 540 or frame.shape[0] == 1080
    assert frame.shape[1] >= 1920


@pytest.mark.parametrize("video_name,snapshot_name", SCENARIOS)
def test_comparative_snapshot_exists_and_valid(video_name: str, snapshot_name: str) -> None:
    """Verify each snapshot JPG file exists, is non-empty, and reads cleanly."""
    snapshot_path = MEDIA_DIR / snapshot_name
    assert snapshot_path.exists(), f"Missing snapshot image: {snapshot_path}"
    assert snapshot_path.stat().st_size > 5_000, f"Snapshot image too small: {snapshot_path}"

    img = cv2.imread(str(snapshot_path))
    assert img is not None, f"OpenCV failed to read snapshot: {snapshot_path}"
    assert img.shape[0] > 0 and img.shape[1] > 0
