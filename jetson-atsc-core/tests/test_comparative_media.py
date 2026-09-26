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
def test_comparative_video_stream_properties(video_name: str, snapshot_name: str) -> None:
    """Verify stream properties: full frame count, framerate, and 1920x540 split-screen format."""
    video_path = MEDIA_DIR / video_name
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"OpenCV failed to open: {video_path}"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    assert w == 1920, f"Expected width 1920 for split screen, got {w}"
    assert h == 540, f"Expected height 540 for split screen, got {h}"
    assert 20.0 <= fps <= 35.0, f"Unexpected fps {fps}"
    assert total_frames >= 700, f"Video appears truncated with only {total_frames} frames"


@pytest.mark.parametrize("video_name,snapshot_name", SCENARIOS)
def test_comparative_snapshot_exists_and_valid(video_name: str, snapshot_name: str) -> None:
    """Verify each snapshot JPG file exists, is non-empty, and reads cleanly."""
    snapshot_path = MEDIA_DIR / snapshot_name
    assert snapshot_path.exists(), f"Missing snapshot image: {snapshot_path}"
    assert snapshot_path.stat().st_size > 20_000, f"Snapshot image too small: {snapshot_path}"

    img = cv2.imread(str(snapshot_path))
    assert img is not None, f"OpenCV failed to read snapshot: {snapshot_path}"
    assert img.shape == (540, 1920, 3), f"Unexpected snapshot shape: {img.shape}"


@pytest.mark.parametrize("video_name,snapshot_name", SCENARIOS)
def test_comparative_event_snapshots_exist(video_name: str, snapshot_name: str) -> None:
    """Verify cycle trigger and peak density snapshots exist for each scenario."""
    stem = snapshot_name.replace(".jpg", "")
    cycle1_path = MEDIA_DIR / f"{stem}_cycle1.jpg"
    peak_path = MEDIA_DIR / f"{stem}_peak.jpg"

    assert cycle1_path.exists(), f"Missing cycle 1 trigger snapshot: {cycle1_path}"
    assert peak_path.exists(), f"Missing peak density snapshot: {peak_path}"

    img_c1 = cv2.imread(str(cycle1_path))
    assert img_c1 is not None and img_c1.shape == (540, 1920, 3)

    img_peak = cv2.imread(str(peak_path))
    assert img_peak is not None and img_peak.shape == (540, 1920, 3)


def test_comparative_media_aliases_exist() -> None:
    """Verify standard convenience aliases resolve to non-empty media."""
    aliases = [
        "compare_night_glare.mp4",
        "compare_rain_wet.mp4",
        "compare_congestion_gridlock.mp4",
        "compare_bandung_pasteur.mp4",
        "compare_bekasi_atcs.mp4",
        "snapshot_night_glare.jpg",
        "snapshot_rain_wet.jpg",
        "snapshot_congestion_gridlock.jpg",
        "snapshot_bandung_pasteur.jpg",
        "snapshot_bekasi_atcs.jpg",
    ]
    for alias in aliases:
        alias_path = MEDIA_DIR / alias
        assert alias_path.exists(), f"Missing alias: {alias_path}"
        assert alias_path.stat().st_size > 1_000, f"Alias target too small: {alias_path}"

