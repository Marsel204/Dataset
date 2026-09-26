#!/usr/bin/env python3
"""Acquisition and transcoding script for ATSC stress-testing and Indonesian datasets.

Acquires and standardizes the following sequences to 1080p (1920x1080), 30fps, H.264:
- data/sample_videos/stress_test/night_glare.mp4 (Night low-light & headlight glare)
- data/sample_videos/stress_test/rain_wet.mp4 (Adverse weather & specular rain reflections)
- data/sample_videos/stress_test/congestion_gridlock.mp4 (Severe arterial congestion & queue spillback)
- data/sample_videos/multi_angle_test/video_bandung_pasteur_h264.mp4 (Dishub Bandung ATCS transcoded to H.264)
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("acquire_stress_datasets")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STRESS_DIR = PROJECT_ROOT / "data" / "sample_videos" / "stress_test"
MULTI_ANGLE_DIR = PROJECT_ROOT / "data" / "sample_videos" / "multi_angle_test"

DATASET_CONFIGS = [
    {
        "name": "night_glare",
        "primary_file": STRESS_DIR / "night_glare.mp4",
        "symlink_file": STRESS_DIR / "stress_night_arterial.mp4",
        "url": "https://www.youtube.com/watch?v=rCgwIsl_wi0",
        "sections": "*00:00:10-00:00:40",
        "fps": 30,
        "description": "Night urban arterial with headlight glare and HDR contrast",
    },
    {
        "name": "rain_wet",
        "primary_file": STRESS_DIR / "rain_wet.mp4",
        "symlink_file": STRESS_DIR / "stress_rain_wet_road.mp4",
        "url": "https://www.youtube.com/watch?v=NOWlexW53Jc",
        "sections": "*00:00:15-00:00:45",
        "fps": 30,
        "description": "Adverse weather with wet pavement and specular road reflections",
    },
    {
        "name": "congestion_gridlock",
        "primary_file": STRESS_DIR / "congestion_gridlock.mp4",
        "symlink_file": STRESS_DIR / "stress_congestion_gridlock.mp4",
        "url": "https://www.youtube.com/watch?v=0z3_rZfBpOA",
        "sections": "*00:00:20-00:00:50",
        "fps": 30,
        "description": "Dense urban arterial congestion with severe queue spillback",
    },
]


def verify_video(video_path: Path, min_duration_sec: float = 25.0) -> bool:
    """Verify video exists, is > 1MB, has 1920x1080 resolution, and decodes in OpenCV."""
    if not video_path.exists():
        logger.error(f"Video does not exist: {video_path}")
        return False

    size_bytes = video_path.stat().st_size
    if size_bytes < 1024 * 1024:
        logger.error(f"Video {video_path} is smaller than 1MB: {size_bytes} bytes")
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video with cv2.VideoCapture: {video_path}")
        return False

    ret, frame = cap.read()
    if not ret or frame is None:
        logger.error(f"Failed to read first frame from {video_path}")
        cap.release()
        return False

    h, w, c = frame.shape
    if (w, h) != (1920, 1080):
        logger.error(f"Resolution mismatch in {video_path}: got {w}x{h}, expected 1920x1080")
        cap.release()
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()

    if duration < min_duration_sec:
        logger.error(f"Duration too short in {video_path}: {duration:.2f}s < {min_duration_sec}s")
        return False

    logger.info(
        f"Verified {video_path.name}: {w}x{h}, {fps:.1f} fps, {int(frame_count)} frames, "
        f"{duration:.2f}s, {size_bytes / (1024*1024):.2f}MB"
    )
    return True


def transcode_bandung_pasteur(force: bool = False) -> bool:
    """Transcode video_bandung_pasteur.mp4 from AV1 to H.264."""
    MULTI_ANGLE_DIR.mkdir(parents=True, exist_ok=True)
    h264_target = MULTI_ANGLE_DIR / "video_bandung_pasteur_h264.mp4"
    canonical_link = MULTI_ANGLE_DIR / "video_bandung_pasteur.mp4"
    av1_backup = MULTI_ANGLE_DIR / "video_bandung_pasteur_av1.mp4"

    if h264_target.exists() and not force:
        if verify_video(h264_target):
            logger.info(f"{h264_target.name} already transcoded and valid.")
            if not canonical_link.exists() or (canonical_link.is_symlink() and not canonical_link.resolve().exists()):
                canonical_link.unlink(missing_ok=True)
                canonical_link.symlink_to(h264_target.name)
            return True

    source_input = av1_backup if av1_backup.exists() else canonical_link
    if not source_input.exists():
        logger.error(f"Cannot find source Bandung Pasteur video at {source_input}")
        return False

    # Backup original AV1 if needed
    if source_input == canonical_link and not av1_backup.exists():
        shutil.copy2(canonical_link, av1_backup)

    logger.info(f"Transcoding {source_input} -> {h264_target} via FFmpeg libx264...")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(source_input),
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(h264_target),
    ]
    subprocess.run(cmd, check=True)

    # Symlink canonical name to h264 target
    if canonical_link.exists() or canonical_link.is_symlink():
        canonical_link.unlink()
    canonical_link.symlink_to(h264_target.name)

    return verify_video(h264_target)


def acquire_dataset(config: dict, force: bool = False) -> bool:
    """Acquire a single stress-test dataset via yt-dlp and ffmpeg."""
    primary_file: Path = config["primary_file"]
    symlink_file: Path = config["symlink_file"]

    if primary_file.exists() and not force:
        if verify_video(primary_file):
            logger.info(f"{primary_file.name} already exists and is valid.")
            if not symlink_file.exists():
                symlink_file.symlink_to(primary_file.name)
            return True

    STRESS_DIR.mkdir(parents=True, exist_ok=True)
    temp_raw = STRESS_DIR / f"{config['name']}_raw.mp4"

    logger.info(f"Downloading {config['name']} from {config['url']} ({config['sections']})...")
    dl_cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--download-sections", config["sections"],
        "-o", str(temp_raw),
        config["url"],
    ]
    subprocess.run(dl_cmd, check=True)

    logger.info(f"Standardizing {config['name']} to 1920x1080 30fps H.264...")
    ff_cmd = [
        "ffmpeg", "-y",
        "-i", str(temp_raw),
        "-vf", "scale=1920:1080",
        "-r", str(config["fps"]),
        "-c:v", "libx264",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(primary_file),
    ]
    subprocess.run(ff_cmd, check=True)
    temp_raw.unlink(missing_ok=True)

    # Establish symlink
    if symlink_file.exists() or symlink_file.is_symlink():
        symlink_file.unlink()
    symlink_file.symlink_to(primary_file.name)

    return verify_video(primary_file)


def main():
    parser = argparse.ArgumentParser(description="Acquire and transcode traffic video datasets.")
    parser.add_argument("--force", action="store_true", help="Force re-acquisition and transcoding.")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing dataset files.")
    args = parser.parse_args()

    success = True

    if not args.verify_only:
        logger.info("--- Step 1: Transcoding Bandung Pasteur (AV1 -> H.264) ---")
        if not transcode_bandung_pasteur(force=args.force):
            success = False

        logger.info("--- Step 2: Acquiring Stress-Test Datasets ---")
        for cfg in DATASET_CONFIGS:
            if not acquire_dataset(cfg, force=args.force):
                success = False
    else:
        logger.info("Running in verification-only mode.")

    logger.info("--- Step 3: Verifying All 5 Target Video Feeds ---")
    all_targets = [
        STRESS_DIR / "night_glare.mp4",
        STRESS_DIR / "rain_wet.mp4",
        STRESS_DIR / "congestion_gridlock.mp4",
        MULTI_ANGLE_DIR / "video_bandung_pasteur_h264.mp4",
        MULTI_ANGLE_DIR / "video_bekasi_atcs.mp4",
    ]

    for target in all_targets:
        if not verify_video(target):
            logger.error(f"FAILED verification for {target}")
            success = False
        else:
            logger.info(f"PASSED verification for {target}")

    if not success:
        logger.error("Dataset acquisition / verification encountered failures.")
        sys.exit(1)
    else:
        logger.info("All 5 dataset video feeds are verified and ready!")
        sys.exit(0)


if __name__ == "__main__":
    main()
