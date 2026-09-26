#!/usr/bin/env python3
"""
Synchronized Side-by-Side Comparative Media Rendering Engine for Jetson ATSC Core.

Renders synchronized split-screen comparison videos and high-resolution snapshots:
- Left panel: Raw CCTV Surveillance Feed (unrectified/unannotated).
- Right panel: AI Perception & ANFIS Actuation (rectified, corridor polygon,
  YOLO11+ByteTrack detections, metric ground vectors, and live HUD telemetry).
- Formats:
  - Video: H.264 encoded MP4 (-c:v libx264 -pix_fmt yuv420p -movflags +faststart)
  - Snapshots: High-resolution JPEG at actuation trigger points and peak traffic density.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.common.types import TrafficSnapshot, ActuationDecision
from src.perception.lens_rectifier import LensRectifier
from src.perception.phase_monitor import OpticalPhaseMonitor
from src.perception.detector_tracker import DetectorTracker, TrackedVehicle
from src.analytics.homography_engine import HomographyEngine
from src.analytics.traffic_metrics import TrafficMetricsExtractor
from src.control.anfis_inference import ANFISInferenceEngine
from src.control.decision_engine import ATSCDecisionEngine


SCENARIO_CATALOG: Dict[str, Dict[str, Any]] = {
    "night_glare": {
        "id": "scenario_1",
        "name": "Scenario 1: Night Glare",
        "title": "Night & Low-Light Urban Arterial with Headlamp Glare",
        "location": "US Airline Highway CCTV (DOT Feed)",
        "source": "data/sample_videos/stress_test/night_glare.mp4",
        "config": "configs/stress_night_arterial.json",
        "output_video": "compare_stress_night_arterial.mp4",
        "output_snapshot": "snapshot_stress_night_arterial.jpg",
        "video_aliases": ["compare_night_glare.mp4"],
        "snapshot_aliases": ["snapshot_night_glare.jpg"],
        "trigger_interval_sec": 15.0,
    },
    "rain_wet": {
        "id": "scenario_2",
        "name": "Scenario 2: Rain & Wet Road",
        "title": "Adverse Weather / Wet Pavement Specular Reflections",
        "location": "Ontario 511 Highway CCTV Feed",
        "source": "data/sample_videos/stress_test/rain_wet.mp4",
        "config": "configs/stress_rain_wet_road.json",
        "output_video": "compare_stress_rain_wet_road.mp4",
        "output_snapshot": "snapshot_stress_rain_wet_road.jpg",
        "video_aliases": ["compare_rain_wet.mp4", "compare_stress_rain_wet.mp4"],
        "snapshot_aliases": ["snapshot_rain_wet.jpg", "snapshot_stress_rain_wet.jpg"],
        "trigger_interval_sec": 15.0,
    },
    "congestion_gridlock": {
        "id": "scenario_3",
        "name": "Scenario 3: Congestion & Gridlock",
        "title": "Dense Urban Arterial Congestion & Queue Spillback",
        "location": "Chicago Urban Arterial Surveillance (Midwest Feed)",
        "source": "data/sample_videos/stress_test/congestion_gridlock.mp4",
        "config": "configs/stress_congestion_gridlock.json",
        "output_video": "compare_stress_congestion_gridlock.mp4",
        "output_snapshot": "snapshot_stress_congestion_gridlock.jpg",
        "video_aliases": ["compare_congestion_gridlock.mp4"],
        "snapshot_aliases": ["snapshot_congestion_gridlock.jpg"],
        "trigger_interval_sec": 15.0,
    },
    "bandung_pasteur": {
        "id": "scenario_4",
        "name": "Scenario 4: Indonesian Bandung Pasteur ATCS",
        "title": "Indonesian Dishub Bandung ATCS — Simpang Pasteur",
        "location": "Simpang Pasteur, Kota Bandung (High MC Density & Mixed Traffic)",
        "source": "data/sample_videos/multi_angle_test/video_bandung_pasteur_h264.mp4",
        "config": "configs/stress_indonesia_bandung_pasteur.json",
        "output_video": "compare_stress_indonesia_bandung_pasteur.mp4",
        "output_snapshot": "snapshot_stress_indonesia_bandung_pasteur.jpg",
        "video_aliases": ["compare_video_bandung_pasteur.mp4", "compare_bandung_pasteur.mp4"],
        "snapshot_aliases": ["snapshot_video_bandung_pasteur.jpg", "snapshot_bandung_pasteur.jpg"],
        "trigger_interval_sec": 15.0,
    },
    "bekasi_atcs": {
        "id": "scenario_5",
        "name": "Scenario 5: Indonesian Bekasi ATCS",
        "title": "Indonesian Dishub Bekasi ATCS — Arterial Corridor",
        "location": "Arterial Intersection ATCS, Kota Bekasi (Mixed Traffic)",
        "source": "data/sample_videos/multi_angle_test/video_bekasi_atcs.mp4",
        "config": "configs/stress_indonesia_bekasi.json",
        "output_video": "compare_stress_indonesia_bekasi.mp4",
        "output_snapshot": "snapshot_stress_indonesia_bekasi.jpg",
        "video_aliases": ["compare_video_bekasi_atcs.mp4", "compare_bekasi_atcs.mp4"],
        "snapshot_aliases": ["snapshot_video_bekasi_atcs.jpg", "snapshot_bekasi_atcs.jpg"],
        "trigger_interval_sec": 15.0,
    },
}


class FFmpegH264Writer:
    """
    Direct pipe video writer sending raw BGR frames to FFmpeg for native H.264
    encoding with web-compatible yuv420p pixel format and faststart flags.
    """

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        crf: int = 22,
        preset: str = "fast",
    ) -> None:
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.proc: Optional[subprocess.Popen] = None
        self.fallback_writer: Optional[cv2.VideoWriter] = None
        self.temp_cv2_path: Optional[str] = None

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", f"{fps:.2f}",
            "-i", "-",
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ]

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception as e:
            print(f"[FFmpegH264Writer] FFmpeg spawn failed: {e}. Falling back to OpenCV VideoWriter.")
            self._init_fallback()

    def _init_fallback(self) -> None:
        self.temp_cv2_path = self.output_path + ".tmp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.fallback_writer = cv2.VideoWriter(
            self.temp_cv2_path,
            fourcc,
            self.fps,
            (self.width, self.height),
        )

    def write(self, frame: np.ndarray) -> None:
        """Writes a BGR frame to the encoder."""
        if self.proc and self.proc.stdin:
            try:
                self.proc.stdin.write(frame.tobytes())
                return
            except Exception as e:
                print(f"[FFmpegH264Writer] Pipe write error: {e}. Switching to OpenCV fallback.")
                self.proc = None
                self._init_fallback()

        if self.fallback_writer:
            self.fallback_writer.write(frame)

    def release(self) -> None:
        """Flushes buffers and finalizes the MP4 file."""
        if self.proc:
            if self.proc.stdin:
                self.proc.stdin.close()
            stderr_out = self.proc.stderr.read() if self.proc.stderr else b""
            self.proc.wait()
            if self.proc.returncode != 0:
                print(f"[FFmpegH264Writer] FFmpeg exited with code {self.proc.returncode}: {stderr_out.decode('utf-8', errors='ignore')}")

        if self.fallback_writer:
            self.fallback_writer.release()
            if self.temp_cv2_path and os.path.exists(self.temp_cv2_path):
                # Transcode intermediate mp4v to standard H.264
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", self.temp_cv2_path,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        self.output_path,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                try:
                    os.remove(self.temp_cv2_path)
                except OSError:
                    pass


class ComparativeMediaRenderer:
    """
    Renders synchronized side-by-side comparative traffic analytics media.
    """

    def __init__(
        self,
        video_path: str,
        config_path: str,
        output_video_path: str,
        output_snapshot_path: str,
        scenario_meta: Optional[Dict[str, Any]] = None,
        panel_width: int = 960,
        panel_height: int = 540,
        trigger_interval_sec: float = 15.0,
        max_frames: Optional[int] = None,
        imgsz: int = 640,
    ) -> None:
        self.video_path = os.path.abspath(video_path)
        self.config_path = os.path.abspath(config_path)
        self.output_video_path = os.path.abspath(output_video_path)
        self.output_snapshot_path = os.path.abspath(output_snapshot_path)
        self.scenario_meta = scenario_meta or {}
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.trigger_interval_sec = trigger_interval_sec
        self.max_frames = max_frames
        self.imgsz = imgsz

        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Input video not found: {self.video_path}")
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        sys_cfg = self.config.get("system", {})
        calib_rel = sys_cfg.get("calibration_path", "configs/rectilinear_calib.npz")
        model_rel = sys_cfg.get("model_path", "models/yolo11s.pt")
        anfis_rel = "models/anfis_sugeno.onnx"

        calib_path = os.path.join(PROJECT_ROOT, calib_rel) if not os.path.isabs(calib_rel) else calib_rel
        model_path = os.path.join(PROJECT_ROOT, model_rel) if not os.path.isabs(model_rel) else model_rel
        anfis_path = os.path.join(PROJECT_ROOT, anfis_rel)

        # Initialize perception, analytics, and control components
        self.rectifier = LensRectifier(calib_path)
        self.phase_monitor = OpticalPhaseMonitor.from_config(self.config)
        self.homography = HomographyEngine.from_config(self.config)
        self.metrics_extractor = TrafficMetricsExtractor.from_config(self.config, self.homography)
        self.detector = DetectorTracker(model_path, conf_threshold=0.28, imgsz=self.imgsz)

        green_bounds = self.config.get("lane_metrics", {}).get("green_time_bounds_sec", [10.0, 120.0])
        self.anfis = ANFISInferenceEngine(
            anfis_path,
            prefer_cuda=False,
            min_green_sec=green_bounds[0],
            max_green_sec=green_bounds[1],
        )
        self.decision_engine = ATSCDecisionEngine(self.anfis)

        # Video ingestion
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.orig_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        self.orig_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

        # Output video dimensions (left + right = 2 * panel_width)
        self.out_width = self.panel_width * 2
        self.out_height = self.panel_height

        self.last_trigger_timestamp: float = 0.0
        self.cycle_count: int = 0
        self.last_decision: Optional[ActuationDecision] = None
        self.last_decision_timestamp: float = -999.0

        # Snapshot candidates
        self.cycle1_frame: Optional[np.ndarray] = None
        self.peak_frame: Optional[np.ndarray] = None
        self.max_density_score: float = -1.0
        self.peak_metrics: Dict[str, Any] = {}

    def render(self) -> Dict[str, Any]:
        """
        Executes end-to-end synchronized side-by-side comparative rendering.
        """
        writer = FFmpegH264Writer(
            output_path=self.output_video_path,
            width=self.out_width,
            height=self.out_height,
            fps=self.fps,
        )

        frame_idx = 0
        latencies: List[float] = []
        fps_records: List[float] = []
        decisions: List[Dict[str, Any]] = []

        scenario_name = self.scenario_meta.get("name", "Stress Scenario")
        location_desc = self.scenario_meta.get("location", "Surveillance Camera Feed")

        print(f"\n[ComparativeMediaRenderer] Rendering {scenario_name}...")
        print(f"  Input  : {self.video_path} ({self.total_frames} frames @ {self.fps:.1f} FPS)")
        print(f"  Output : {self.output_video_path} ({self.out_width}x{self.out_height} H.264)")

        t_render_start = time.time()

        while True:
            if self.max_frames and frame_idx >= self.max_frames:
                break

            ret, raw_frame = self.cap.read()
            if not ret or raw_frame is None:
                break

            frame_idx += 1
            timestamp = frame_idx / self.fps
            t_frame_start = time.perf_counter()

            # 1. Pipeline Execution
            rectified = self.rectifier.rectify(raw_frame)
            yellow_triggered = self.phase_monitor.process_frame(rectified, timestamp)
            tracks = self.detector.track([rectified], timestamp=timestamp)
            vehicles = tracks[0] if tracks else []
            self.detector.update_velocities(vehicles, self.homography.pixel_to_ground, timestamp=timestamp)

            raw_metrics = self.metrics_extractor.extract_approach_metrics(vehicles)
            snapshot = TrafficSnapshot.from_dict(raw_metrics, timestamp=timestamp)

            # Cycle Trigger Evaluation
            cycle_triggered = yellow_triggered
            if not cycle_triggered and self.trigger_interval_sec > 0:
                if (timestamp - self.last_trigger_timestamp) >= self.trigger_interval_sec and timestamp >= self.trigger_interval_sec:
                    cycle_triggered = True
                    self.last_trigger_timestamp = timestamp

            if cycle_triggered:
                self.cycle_count += 1
                self.last_decision = self.decision_engine.evaluate(snapshot, self.cycle_count)
                self.last_decision_timestamp = timestamp
                decisions.append({
                    "cycle_id": self.cycle_count,
                    "frame": frame_idx,
                    "timestamp": round(timestamp, 2),
                    "green_seconds": self.last_decision.green_seconds,
                    "inputs": [snapshot.v_w_pcu, snapshot.queue_meters, snapshot.occupancy_pct],
                    "inference_latency_ms": self.last_decision.inference_latency_ms,
                })

            t_frame_end = time.perf_counter()
            frame_lat_ms = (t_frame_end - t_frame_start) * 1000.0
            latencies.append(frame_lat_ms)
            current_fps = 1000.0 / frame_lat_ms if frame_lat_ms > 0 else self.fps
            fps_records.append(current_fps)

            # 2. Render Left Panel (Raw CCTV Feed)
            left_panel = cv2.resize(raw_frame, (self.panel_width, self.panel_height), interpolation=cv2.INTER_AREA)
            self._render_left_hud(left_panel, frame_idx, timestamp, scenario_name, location_desc)

            # 3. Render Right Panel (AI Annotated & Corridor)
            annotated_full = self._render_right_annotations(rectified, vehicles)
            right_panel = cv2.resize(annotated_full, (self.panel_width, self.panel_height), interpolation=cv2.INTER_AREA)
            self._render_right_hud(
                right_panel,
                frame_idx,
                timestamp,
                snapshot,
                frame_lat_ms,
                current_fps,
            )

            # 4. Compose Split-Screen with Divider
            # Embed 2-pixel divider inside left panel's right edge for seamless geometry
            left_panel[:, -2:] = (0, 215, 255)  # Gold vertical divider line
            split_frame = np.hstack([left_panel, right_panel])

            writer.write(split_frame)

            # 5. Snapshot Tracking (Actuation Trigger Point & Peak Demand)
            if self.cycle_count == 1 and self.cycle1_frame is None and (timestamp - self.last_decision_timestamp) < 0.2:
                self.cycle1_frame = split_frame.copy()

            density_score = snapshot.v_w_pcu + (snapshot.occupancy_pct * 0.5) + (snapshot.queue_meters * 0.2)
            if density_score > self.max_density_score:
                self.max_density_score = density_score
                self.peak_frame = split_frame.copy()
                self.peak_metrics = {
                    "frame": frame_idx,
                    "timestamp": round(timestamp, 2),
                    "v_w": snapshot.v_w_pcu,
                    "q": snapshot.queue_meters,
                    "l": snapshot.occupancy_pct,
                }

            if frame_idx % 150 == 0 or frame_idx == self.total_frames:
                print(f"  Frame {frame_idx:4d}/{self.total_frames} | Latency: {frame_lat_ms:5.1f}ms | FPS: {current_fps:4.1f} | Cycles: {self.cycle_count}")

        writer.release()
        self.cap.release()

        render_duration = time.time() - t_render_start
        mean_fps = frame_idx / render_duration if render_duration > 0 else 0.0

        # 6. Save Snapshots
        primary_snapshot = self.peak_frame if self.peak_frame is not None else self.cycle1_frame
        if primary_snapshot is None and 'split_frame' in locals():
            primary_snapshot = split_frame

        if primary_snapshot is not None:
            os.makedirs(os.path.dirname(self.output_snapshot_path), exist_ok=True)
            cv2.imwrite(self.output_snapshot_path, primary_snapshot, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"  Primary Snapshot saved: {self.output_snapshot_path}")

            # Also save trigger-specific snapshots if distinct
            base_dir = os.path.dirname(self.output_snapshot_path)
            base_stem = os.path.splitext(os.path.basename(self.output_snapshot_path))[0]

            if self.cycle1_frame is not None:
                c1_path = os.path.join(base_dir, f"{base_stem}_cycle1.jpg")
                cv2.imwrite(c1_path, self.cycle1_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            if self.peak_frame is not None:
                peak_path = os.path.join(base_dir, f"{base_stem}_peak.jpg")
                cv2.imwrite(peak_path, self.peak_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # 7. Create Aliases / Symlinks for seamless cross-referencing
        self._create_aliases()

        # 8. Post-Render Verification
        file_size = os.path.getsize(self.output_video_path) if os.path.exists(self.output_video_path) else 0
        verify_cap = cv2.VideoCapture(self.output_video_path)
        is_decodable = verify_cap.isOpened()
        v_read_ok, test_frame = verify_cap.read() if is_decodable else (False, None)
        verify_cap.release()

        print(f"  Completed in {render_duration:.2f}s ({mean_fps:.1f} FPS)")
        print(f"  Video Size: {file_size / (1024 * 1024):.2f} MB | Decodable: {v_read_ok}")

        return {
            "scenario": self.scenario_meta.get("name", "Stress Scenario"),
            "frames_processed": frame_idx,
            "render_duration_sec": round(render_duration, 2),
            "mean_processing_fps": round(mean_fps, 1),
            "video_path": self.output_video_path,
            "video_size_bytes": file_size,
            "video_decodable": v_read_ok,
            "snapshot_path": self.output_snapshot_path,
            "cycles_triggered": len(decisions),
            "decisions": decisions,
            "peak_metrics": self.peak_metrics,
        }

    def _render_left_hud(
        self,
        panel: np.ndarray,
        frame_idx: int,
        timestamp: float,
        scenario_name: str,
        location_desc: str,
    ) -> None:
        """Renders header and footer HUD onto the raw feed panel."""
        h, w = panel.shape[:2]

        # Top Header Bar
        cv2.rectangle(panel, (0, 0), (w, 38), (20, 20, 20), -1)
        cv2.putText(
            panel,
            f"RAW CCTV FEED: {scenario_name}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        time_tag = f"FRAME {frame_idx:04d} | t={timestamp:5.2f}s"
        cv2.putText(
            panel,
            time_tag,
            (w - 240, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # Bottom Footer Bar
        cv2.rectangle(panel, (0, h - 26), (w, h), (20, 20, 20), -1)
        cv2.putText(
            panel,
            f"Source: {location_desc} | Optical Baseline",
            (12, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    def _render_right_annotations(
        self,
        rectified: np.ndarray,
        vehicles: List[TrackedVehicle],
    ) -> np.ndarray:
        """Renders rich detection boxes and corridor overlay on 1080p frame."""
        vis = rectified.copy()

        # Homography Corridor Overlay
        self.homography.draw_corridor_overlay(vis)

        # Vehicle Bounding Boxes & Ground Vectors
        for v in vehicles:
            x1, y1, x2, y2 = [int(b) for b in v.bbox]
            color = (0, 0, 255) if v.is_stopped else (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Ground position tag
            g_str = f"({v.ground_pos[0]:.1f}m,{v.ground_pos[1]:.1f}m)" if v.ground_pos else ""
            status_tag = "STOPPED" if v.is_stopped else f"{v.velocity_mps:.1f}m/s"
            label = f"#{v.track_id} {v.class_name} {status_tag} {g_str}"
            cv2.putText(
                vis,
                label,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

            # Contact point dot
            bx, by = int(v.bottom_center[0]), int(v.bottom_center[1])
            cv2.circle(vis, (bx, by), 5, (0, 255, 255), -1)

        return vis

    def _render_right_hud(
        self,
        panel: np.ndarray,
        frame_idx: int,
        timestamp: float,
        snap: TrafficSnapshot,
        latency_ms: float,
        fps: float,
    ) -> None:
        """Renders live telemetry, ANFIS actuation callout, and edge specs."""
        h, w = panel.shape[:2]

        # Top Header Bar
        cv2.rectangle(panel, (0, 0), (w, 38), (20, 20, 20), -1)
        cv2.putText(
            panel,
            "AI PERCEPTION & ANFIS ACTUATION",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (50, 255, 50),
            2,
            cv2.LINE_AA,
        )

        metrics_str = f"V_w:{snap.v_w_pcu:4.1f} PCU | Q:{snap.queue_meters:4.1f}m | L:{snap.occupancy_pct:4.1f}%"
        cv2.putText(
            panel,
            metrics_str,
            (w - 340, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Actuation Callout Banner (persists for 3.5s after trigger)
        time_since_actuation = timestamp - self.last_decision_timestamp
        if self.last_decision and time_since_actuation < 3.5:
            # Highlight Banner
            bar_color = (0, 90, 0)  # Forest Green
            cv2.rectangle(panel, (10, 44), (w - 10, 74), bar_color, -1)
            cv2.rectangle(panel, (10, 44), (w - 10, 74), (0, 255, 255), 1)
            act_text = (
                f"⚡ CYCLE #{self.last_decision.cycle_id} ACTUATION: "
                f"{self.last_decision.green_seconds:.1f}s GREEN (t_ANFIS) | Lat: {self.last_decision.inference_latency_ms:.2f}ms"
            )
            cv2.putText(
                panel,
                act_text,
                (18, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            # Baseline Status
            status_text = "ANFIS STATE: MONITORING TRAFFIC | BOUNDS: [10.0s, 120.0s]"
            cv2.putText(
                panel,
                status_text,
                (14, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (170, 170, 170),
                1,
                cv2.LINE_AA,
            )

        # Bottom Footer Bar
        cv2.rectangle(panel, (0, h - 26), (w, h), (20, 20, 20), -1)
        footer_text = f"Jetson Orin Edge | E2E Latency: {latency_ms:4.1f}ms ({fps:4.1f} FPS) | Sugeno 27-Rule"
        cv2.putText(
            panel,
            footer_text,
            (12, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    def _create_aliases(self) -> None:
        """Creates symlinks / copies for requested alias filenames."""
        out_dir = os.path.dirname(self.output_video_path)
        base_video = os.path.basename(self.output_video_path)
        base_snapshot = os.path.basename(self.output_snapshot_path)

        for alias in self.scenario_meta.get("video_aliases", []):
            alias_path = os.path.join(out_dir, alias)
            if not os.path.exists(alias_path) and os.path.exists(self.output_video_path):
                try:
                    os.symlink(base_video, alias_path)
                except OSError:
                    shutil.copy2(self.output_video_path, alias_path)

        for alias in self.scenario_meta.get("snapshot_aliases", []):
            alias_path = os.path.join(out_dir, alias)
            if not os.path.exists(alias_path) and os.path.exists(self.output_snapshot_path):
                try:
                    os.symlink(base_snapshot, alias_path)
                except OSError:
                    shutil.copy2(self.output_snapshot_path, alias_path)


def render_scenario(
    scenario_key: str,
    output_dir: str = "data/comparative_media",
    max_frames: Optional[int] = None,
    panel_width: int = 960,
    panel_height: int = 540,
) -> Dict[str, Any]:
    """Renders a single scenario from the catalog."""
    if scenario_key not in SCENARIO_CATALOG:
        raise KeyError(f"Unknown scenario key: {scenario_key}. Choose from: {list(SCENARIO_CATALOG.keys())}")

    meta = SCENARIO_CATALOG[scenario_key]
    out_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(out_dir, exist_ok=True)

    video_src = os.path.join(PROJECT_ROOT, meta["source"])
    config_src = os.path.join(PROJECT_ROOT, meta["config"])
    out_vid = os.path.join(out_dir, meta["output_video"])
    out_snap = os.path.join(out_dir, meta["output_snapshot"])

    renderer = ComparativeMediaRenderer(
        video_path=video_src,
        config_path=config_src,
        output_video_path=out_vid,
        output_snapshot_path=out_snap,
        scenario_meta=meta,
        panel_width=panel_width,
        panel_height=panel_height,
        trigger_interval_sec=meta.get("trigger_interval_sec", 15.0),
        max_frames=max_frames,
    )
    return renderer.render()


def render_all_scenarios(
    output_dir: str = "data/comparative_media",
    max_frames: Optional[int] = None,
    panel_width: int = 960,
    panel_height: int = 540,
) -> List[Dict[str, Any]]:
    """Renders all 5 scenarios sequentially."""
    results = []
    print("\n==================================================================")
    print("      ATSC COMPARATIVE MEDIA RENDERING SUITE (5 SCENARIOS)        ")
    print("==================================================================")
    t0 = time.time()
    for key in SCENARIO_CATALOG:
        res = render_scenario(
            scenario_key=key,
            output_dir=output_dir,
            max_frames=max_frames,
            panel_width=panel_width,
            panel_height=panel_height,
        )
        results.append(res)

    total_time = time.time() - t0
    print("\n==================================================================")
    print(f"  All 5 scenarios rendered successfully in {total_time:.2f}s!")
    print("==================================================================\n")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronized Comparative Media Renderer")
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["all"] + list(SCENARIO_CATALOG.keys()),
        help="Scenario to render ('all' or scenario key)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/comparative_media",
        help="Output directory for MP4 videos and snapshots",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional maximum frames to process per scenario",
    )
    parser.add_argument(
        "--panel_width",
        type=int,
        default=960,
        help="Width of each panel in split-screen (default 960 -> 1920 total width)",
    )
    parser.add_argument(
        "--panel_height",
        type=int,
        default=540,
        help="Height of split-screen panels (default 540)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scenario == "all":
        render_all_scenarios(
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            panel_width=args.panel_width,
            panel_height=args.panel_height,
        )
    else:
        render_scenario(
            scenario_key=args.scenario,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            panel_width=args.panel_width,
            panel_height=args.panel_height,
        )


if __name__ == "__main__":
    main()
