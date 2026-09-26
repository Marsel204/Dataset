#!/usr/bin/env python3
"""Milestone 1 Verification Script.

Asserts:
1. All 5 video files exist and are > 1MB.
2. OpenCV cv2.VideoCapture successfully opens and reads every video without error.
3. Resolution is exactly 1920x1080.
4. Duration is >= 25 seconds for all videos.
"""

import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGET_VIDEOS = [
    PROJECT_ROOT / "data/sample_videos/stress_test/night_glare.mp4",
    PROJECT_ROOT / "data/sample_videos/stress_test/rain_wet.mp4",
    PROJECT_ROOT / "data/sample_videos/stress_test/congestion_gridlock.mp4",
    PROJECT_ROOT / "data/sample_videos/multi_angle_test/video_bandung_pasteur_h264.mp4",
    PROJECT_ROOT / "data/sample_videos/multi_angle_test/video_bekasi_atcs.mp4",
]

# Canonical path alias verification
ALIAS_VIDEOS = [
    (PROJECT_ROOT / "data/sample_videos/multi_angle_test/video_bandung_pasteur.mp4", "Canonical Bandung Pasteur alias"),
    (PROJECT_ROOT / "data/sample_videos/stress_test/stress_night_arterial.mp4", "Night arterial alias"),
    (PROJECT_ROOT / "data/sample_videos/stress_test/stress_rain_wet_road.mp4", "Rain wet road alias"),
    (PROJECT_ROOT / "data/sample_videos/stress_test/stress_congestion_gridlock.mp4", "Congestion gridlock alias"),
]


def test_video(path: Path) -> dict:
    assert path.exists(), f"Video file does not exist: {path}"
    size_bytes = path.stat().st_size
    assert size_bytes > 1024 * 1024, f"File {path.name} is too small: {size_bytes} <= 1MB"

    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"OpenCV failed to open: {path}"

    ret, frame = cap.read()
    assert ret, f"Failed to read first frame from: {path}"
    assert frame is not None, f"Frame is None for: {path}"

    h, w, c = frame.shape
    assert (w, h) == (1920, 1080), f"Resolution mismatch for {path.name}: expected (1920, 1080), got ({w}, {h})"

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    assert fps > 0, f"Invalid FPS {fps} for {path.name}"
    duration = frame_count / fps
    assert duration >= 25.0, f"Duration {duration:.2f}s is less than 25.0s for {path.name}"

    # Also test reading a frame at 50% and 90% timeline to guarantee frame continuity
    for ratio in [0.5, 0.9]:
        target_frame = int(frame_count * ratio)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret_mid, frame_mid = cap.read()
        assert ret_mid, f"Failed to seek and read frame at index {target_frame} ({ratio*100}%) from {path.name}"
        assert frame_mid.shape == (1080, 1920, 3), f"Frame shape mismatch at index {target_frame} for {path.name}"

    cap.release()
    return {
        "path": str(path),
        "name": path.name,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "resolution": f"{w}x{h}",
        "fps": round(fps, 2),
        "frames": int(frame_count),
        "duration_sec": round(duration, 2),
    }


def main():
    print("==================================================================")
    print("RUNNING MILESTONE 1 VIDEO VERIFICATION SUITE")
    print("==================================================================")
    all_passed = True
    results = []

    print("\n[1/2] Verifying 5 Mandatory Benchmark Video Streams:")
    for video_path in TARGET_VIDEOS:
        try:
            info = test_video(video_path)
            results.append(info)
            print(f"  ✓ PASS: {info['name']} | {info['resolution']} @ {info['fps']}fps | "
                  f"{info['duration_sec']}s ({info['frames']} frames) | {info['size_mb']} MB")
        except AssertionError as e:
            print(f"  ✗ FAIL: {video_path.name}: {e}")
            all_passed = False

    print("\n[2/2] Verifying Symbolic Aliases & Backward Compatibility:")
    for alias_path, desc in ALIAS_VIDEOS:
        try:
            info = test_video(alias_path)
            print(f"  ✓ PASS: {desc} ({alias_path.name}) -> {info['resolution']} @ {info['fps']}fps")
        except AssertionError as e:
            print(f"  ✗ FAIL: {desc} ({alias_path.name}): {e}")
            all_passed = False

    print("\n==================================================================")
    if all_passed:
        print("ALL MILESTONE 1 VIDEO VERIFICATION CHECKS PASSED!")
        print("==================================================================")
        sys.exit(0)
    else:
        print("MILESTONE 1 VIDEO VERIFICATION FAILED!")
        print("==================================================================")
        sys.exit(1)


if __name__ == "__main__":
    main()
