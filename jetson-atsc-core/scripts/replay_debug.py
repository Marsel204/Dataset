#!/usr/bin/env python3
"""
Deterministic Single-Threaded Replay Debugger for ATSC Pipeline.

Provides interactive and headless frame-by-step execution on recorded footage
without live camera hardware, serial interfaces, or multithreading race conditions.
Allows precise inspection of optics unwarping, vehicle detections, IPM ground coordinates,
optical phase transitions, and ANFIS green allocations.

Interactive Controls (GUI Mode):
  [SPACE]  : Step 1 frame forward
  [c]      : Run continuously until next yellow phase trigger or EOF
  [p]      : Print detailed JSON state of current frame to console
  [s]      : Save snapshot image of current frame with HUD to disk
  [q]/[ESC]: Quit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

# Ensure project root is in sys.path
CORE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)

from src.common.types import TrafficSnapshot, ActuationDecision
from src.perception.lens_rectifier import LensRectifier
from src.perception.phase_monitor import OpticalPhaseMonitor
from src.perception.detector_tracker import DetectorTracker, TrackedVehicle
from src.analytics.homography_engine import HomographyEngine
from src.analytics.traffic_metrics import TrafficMetricsExtractor
from src.control.anfis_inference import ANFISInferenceEngine
from src.control.decision_engine import ATSCDecisionEngine


class ReplayDebugger:
    """
    Synchronous, deterministic debugging engine for ATSC perception and control.
    """

    def __init__(
        self,
        video_path: str,
        config_path: str,
        calib_path: str,
        model_path: str,
        anfis_path: str,
        headless: bool = False,
        step_mode: bool = True,
        max_frames: Optional[int] = None,
        trace_path: Optional[str] = None,
        imgsz: int = 640,
    ) -> None:
        self.video_path = os.path.abspath(video_path)
        self.headless = headless
        self.step_mode = step_mode
        self.max_frames = max_frames
        self.trace_path = os.path.abspath(trace_path) if trace_path else None
        self.imgsz = imgsz

        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        # Load configurations
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Initialize core components
        print(f"[ReplayDebugger] Loading Lens Rectifier from: {calib_path}")
        self.rectifier = LensRectifier(calib_path)

        print("[ReplayDebugger] Initializing Optical Phase Monitor...")
        self.phase_monitor = OpticalPhaseMonitor.from_config(self.config)

        print("[ReplayDebugger] Initializing Homography Engine...")
        self.homography = HomographyEngine.from_config(self.config)

        print("[ReplayDebugger] Initializing Traffic Metrics Extractor...")
        self.metrics_extractor = TrafficMetricsExtractor.from_config(self.config, self.homography)

        print(f"[ReplayDebugger] Loading Detector from: {model_path}...")
        self.detector = DetectorTracker(model_path, conf_threshold=0.28, imgsz=self.imgsz)

        print(f"[ReplayDebugger] Loading ANFIS Engine from: {anfis_path}...")
        self.anfis = ANFISInferenceEngine(
            anfis_path,
            prefer_cuda=False,  # CPU is robust for local debugging
            min_green_sec=self.config.get("lane_metrics", {}).get("green_time_bounds_sec", [10.0, 120.0])[0],
            max_green_sec=self.config.get("lane_metrics", {}).get("green_time_bounds_sec", [10.0, 120.0])[1],
        )
        self.decision_engine = ATSCDecisionEngine(self.anfis)

        # Video capture
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video capture: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.frame_idx = 0
        self.cycle_count = 0
        self.last_decision: Optional[ActuationDecision] = None
        self.current_snapshot: Optional[TrafficSnapshot] = None
        self.current_vehicles: List[TrackedVehicle] = []
        self.trace_file = None

        if self.trace_path:
            os.makedirs(os.path.dirname(self.trace_path), exist_ok=True)
            self.trace_file = open(self.trace_path, "w", encoding="utf-8")

    def run(self) -> None:
        """Executes the replay loop."""
        print("\n==================================================================")
        print("          ATSC DETERMINISTIC REPLAY DEBUGGER                      ")
        print("==================================================================")
        print(f"Video Source : {self.video_path} ({self.total_frames} frames @ {self.fps:.1f} FPS)")
        print(f"Headless     : {self.headless}")
        print(f"Initial Mode : {'STEP (Paused on each frame)' if self.step_mode else 'CONTINUOUS'}")
        if self.trace_path:
            print(f"Trace Output : {self.trace_path}")
        print("==================================================================\n")

        running = True
        while running:
            if self.max_frames and self.frame_idx >= self.max_frames:
                print(f"[ReplayDebugger] Reached max frame limit ({self.max_frames}). Stopping.")
                break

            ret, raw_frame = self.cap.read()
            if not ret or raw_frame is None:
                print("[ReplayDebugger] End of video stream reached.")
                break

            self.frame_idx += 1
            timestamp = self.frame_idx / self.fps

            # 1. Optics Rectification
            rectified = self.rectifier.rectify(raw_frame)

            # 2. Optical Yellow Detection
            yellow_triggered = self.phase_monitor.process_frame(rectified, timestamp)

            # 3. Detection & Tracking
            tracks = self.detector.track([rectified], timestamp=timestamp)
            vehicles = tracks[0] if tracks else []
            self.detector.update_velocities(vehicles, self.homography.pixel_to_ground)
            self.current_vehicles = vehicles

            # 4. PKJI Traffic Metrics
            raw_metrics = self.metrics_extractor.extract_approach_metrics(vehicles)
            self.current_snapshot = TrafficSnapshot.from_dict(raw_metrics, timestamp=timestamp)

            # 5. ANFIS Green Time on Yellow Trigger
            if yellow_triggered:
                self.cycle_count += 1
                self.last_decision = self.decision_engine.evaluate(self.current_snapshot, self.cycle_count)
                print(f"\n⚡ [CYCLE #{self.cycle_count} TRIGGERED at Frame {self.frame_idx} | t={timestamp:.2f}s]")
                print(f"   V_w={self.current_snapshot.v_w_pcu:.2f} PCU | Q={self.current_snapshot.queue_meters:.1f}m | L={self.current_snapshot.occupancy_pct:.1f}%")
                print(f"   -> Allocated Green: {self.last_decision.green_seconds:.1f}s (Latency: {self.last_decision.inference_latency_ms:.2f}ms)\n")

                # If running continuously, pause on yellow trigger for user inspection
                if not self.headless:
                    self.step_mode = True

            # 6. Optional Trace Logging
            if self.trace_file:
                record = {
                    "frame": self.frame_idx,
                    "timestamp": round(timestamp, 3),
                    "yellow_active": self.phase_monitor.is_yellow_active,
                    "yellow_ratio": round(self.phase_monitor.last_active_ratio, 3),
                    "yellow_triggered": yellow_triggered,
                    "traffic": self.current_snapshot.to_dict(),
                    "vehicles": [
                        {
                            "id": v.track_id,
                            "class": v.class_name,
                            "speed_mps": round(v.velocity_mps, 2),
                            "stopped": v.is_stopped,
                            "ground": [round(c, 2) for c in v.ground_pos] if v.ground_pos else None,
                        }
                        for v in vehicles
                    ],
                }
                if yellow_triggered and self.last_decision:
                    record["decision"] = self.last_decision.to_dict()
                self.trace_file.write(json.dumps(record) + "\n")

            # 7. Rendering & User Interaction
            if not self.headless:
                display = self._render(rectified, vehicles)
                cv2.imshow("ATSC Replay Debugger", display)

                wait_time = 0 if self.step_mode else max(1, int(1000.0 / self.fps))
                key = cv2.waitKey(wait_time) & 0xFF

                if key == ord("q") or key == 27:  # Quit
                    print("[ReplayDebugger] Exiting on user command.")
                    break
                elif key == ord(" "):  # Step 1 frame
                    self.step_mode = True
                elif key == ord("c"):  # Continuous run
                    self.step_mode = False
                elif key == ord("p"):  # Print frame state
                    self._print_state()
                elif key == ord("s"):  # Save snapshot
                    shot_path = f"debug_frame_{self.frame_idx:05d}.jpg"
                    cv2.imwrite(shot_path, display)
                    print(f"[ReplayDebugger] Saved snapshot to: {shot_path}")

        self._cleanup()

    def _render(self, frame: np.ndarray, vehicles: List[TrackedVehicle]) -> np.ndarray:
        """Renders rich debugging HUD overlay onto frame."""
        vis = frame.copy()

        # Homography approach corridor polygon
        self.homography.draw_corridor_overlay(vis)

        # Optical Yellow Phase Monitor ROI & Status
        self.phase_monitor.draw_overlay(vis)

        # Tracked Vehicles with Metric Vectors
        for v in vehicles:
            x1, y1, x2, y2 = [int(b) for b in v.bbox]
            color = (0, 0, 255) if v.is_stopped else (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Ground position tag
            g_str = f"({v.ground_pos[0]:.1f}m,{v.ground_pos[1]:.1f}m)" if v.ground_pos else ""
            label = f"#{v.track_id} {v.class_name} {v.velocity_mps:.1f}m/s {g_str}"
            cv2.putText(vis, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

            # Contact point dot
            bx, by = int(v.bottom_center[0]), int(v.bottom_center[1])
            cv2.circle(vis, (bx, by), 4, (0, 255, 255), -1)

        # HUD Top Banner
        snap = self.current_snapshot
        status_bar = f"FRAME: {self.frame_idx}/{self.total_frames} | V_w: {snap.v_w_pcu:.2f} PCU | Q: {snap.queue_meters:.1f}m | L: {snap.occupancy_pct:.1f}% | MODE: {'STEP' if self.step_mode else 'RUN'}"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (20, 20, 20), -1)
        cv2.putText(vis, status_bar, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        # ANFIS Decision Overlay
        if self.last_decision:
            dec_bar = f"LAST ANFIS: {self.last_decision.green_seconds:.1f}s green (Cycle #{self.last_decision.cycle_id})"
            cv2.putText(vis, dec_bar, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 200, 0), 2)

        # Bottom Controls Guide
        help_bar = "[SPACE]: Step | [c]: Continuous | [p]: Dump State | [s]: Screenshot | [q]: Quit"
        cv2.rectangle(vis, (0, vis.shape[0] - 30), (vis.shape[1], vis.shape[0]), (20, 20, 20), -1)
        cv2.putText(vis, help_bar, (15, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

        return vis

    def _print_state(self) -> None:
        """Prints formatted JSON diagnostic state of the current frame to stdout."""
        snap = self.current_snapshot.to_dict() if self.current_snapshot else {}
        print(f"\n--- Frame {self.frame_idx} State Dump ---")
        print(json.dumps({
            "frame": self.frame_idx,
            "yellow_monitor": {
                "active": self.phase_monitor.is_yellow_active,
                "active_ratio": round(self.phase_monitor.last_active_ratio, 3),
            },
            "traffic_snapshot": snap,
            "tracked_vehicles": [
                {
                    "track_id": v.track_id,
                    "class": v.class_name,
                    "speed_mps": round(v.velocity_mps, 2),
                    "is_stopped": v.is_stopped,
                    "bottom_center_pixel": v.bottom_center,
                    "ground_pos_m": v.ground_pos,
                }
                for v in self.current_vehicles
            ]
        }, indent=2))
        print("---------------------------------\n")

    def _cleanup(self) -> None:
        """Releases video and output file handles."""
        self.cap.release()
        if self.trace_file:
            self.trace_file.close()
        if not self.headless:
            cv2.destroyAllWindows()
        print(f"[ReplayDebugger] Cleanup complete. Processed {self.frame_idx} frames, triggered {self.cycle_count} cycles.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ATSC Deterministic Replay Debugger")
    parser.add_argument("--source", type=str, required=True, help="Path to input MP4 video file")
    parser.add_argument("--config", type=str, default=os.path.join(CORE_ROOT, "configs/intersection_roi.json"), help="Path to intersection ROI config")
    parser.add_argument("--calib", type=str, default=os.path.join(CORE_ROOT, "configs/brica_fisheye_calib.npz"), help="Path to fisheye calibration npz")
    parser.add_argument("--model", type=str, default=os.path.join(CORE_ROOT, "models/Final.pt"), help="Path to YOLO weights (.pt or .engine)")
    parser.add_argument("--anfis", type=str, default=os.path.join(CORE_ROOT, "models/anfis_sugeno.onnx"), help="Path to ANFIS ONNX model")
    parser.add_argument("--headless", action="store_true", help="Run without graphical display")
    parser.add_argument("--step", action="store_true", default=False, help="Start in paused step-by-step mode")
    parser.add_argument("--max_frames", type=int, default=None, help="Stop after processing N frames")
    parser.add_argument("--export_trace", type=str, default=None, help="Path to save frame-by-frame JSONL trace")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference resolution")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    debugger = ReplayDebugger(
        video_path=args.source,
        config_path=args.config,
        calib_path=args.calib,
        model_path=args.model,
        anfis_path=args.anfis,
        headless=args.headless,
        step_mode=args.step,
        max_frames=args.max_frames,
        trace_path=args.export_trace,
        imgsz=args.imgsz,
    )
    debugger.run()


if __name__ == "__main__":
    main()
