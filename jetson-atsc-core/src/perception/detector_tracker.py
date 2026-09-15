"""
Shared-Engine YOLO11 Batch Inference & Multi-Object Tracking Engine.

Supports:
- Batched TensorRT FP16 engine (yolo11s_fp16.engine) on Jetson Orin Nano
- Native PyTorch CUDA/CPU fallback (Final.pt)
- Concurrent dual-camera batch tracking (frame_sys + frame_acq) in a single forward pass
- ByteTrack integration with track trajectory and ground velocity estimation
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


# Standard PKJI vehicle categories
CLASS_NAMES = {
    0: "motorcycle",
    1: "car",
    2: "bus",
    3: "truck",
}


class TrackedVehicle:
    """Represents a single tracked vehicle in rectified pixel and metric space."""

    def __init__(
        self,
        track_id: int,
        class_id: int,
        class_name: str,
        bbox: List[float],
        confidence: float,
        timestamp: float,
    ) -> None:
        self.track_id = track_id
        self.class_id = class_id
        self.class_name = class_name
        self.bbox = [float(b) for b in bbox]  # [x1, y1, x2, y2]
        self.confidence = float(confidence)
        self.timestamp = timestamp

        # Contact point: center of bottom edge
        self.bottom_center = (
            float((self.bbox[0] + self.bbox[2]) / 2.0),
            float(self.bbox[3]),
        )

        # Ground plane coordinates [X_m, Y_m] (set by homography engine)
        self.ground_pos: Optional[Tuple[float, float]] = None
        self.velocity_mps: float = 0.0

    @property
    def is_stopped(self) -> bool:
        """Determines if vehicle is queued / stationary (speed <= 1.0 m/s)."""
        return self.velocity_mps <= 1.0


class DetectorTracker:
    """
    Batched vehicle detection and ByteTrack tracking system.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.50,
        imgsz: int = 640,
        device: str = "cuda:0",
    ) -> None:
        """
        Args:
            model_path: Path to TensorRT engine or PyTorch .pt weights.
            conf_threshold: Minimum detection confidence threshold.
            iou_threshold: NMS IoU threshold.
            imgsz: Input resolution size.
            device: 'cuda:0' or 'cpu'.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device

        self.is_tensorrt: bool = str(model_path).endswith(".engine")
        self.model = None

        # Track history for velocity calculation: {track_id: list of (timestamp, ground_x, ground_y)}
        self.track_trajectories: Dict[int, List[Tuple[float, float, float]]] = {}
        self.max_history_len = 10

        self._load_model()

    def _load_model(self) -> None:
        """Loads Ultralytics YOLO model or TensorRT engine."""
        from ultralytics import YOLO

        if not os.path.exists(self.model_path):
            # If specified engine does not exist, check for Final.pt fallback
            alt_pt = os.path.join(os.path.dirname(self.model_path), "Final.pt")
            if os.path.exists(alt_pt):
                print(f"[Detector] Engine '{self.model_path}' not found. Falling back to: {alt_pt}")
                self.model_path = alt_pt
                self.is_tensorrt = False
            else:
                raise FileNotFoundError(f"Neither model nor fallback found at: {self.model_path}")

        print(f"[Detector] Initializing YOLO from: {self.model_path} (TensorRT: {self.is_tensorrt})")
        self.model = YOLO(self.model_path, task="detect")

    def track(
        self,
        frames: List[np.ndarray],
        timestamp: Optional[float] = None,
    ) -> List[List[TrackedVehicle]]:
        """
        Performs batched inference and multi-object tracking across 1 or 2 frames.

        Args:
            frames: List of 1 or 2 BGR images (e.g. [frame_sys] or [frame_sys, frame_acq]).
            timestamp: Ingestion timestamp.

        Returns:
            List of tracked vehicle lists, one list per input frame.
        """
        if timestamp is None:
            timestamp = time.time()

        if not frames:
            return []

        # Batched inference via Ultralytics track
        try:
            results = self.model.track(
                source=frames,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )
        except Exception as e:
            # Fallback if tracker config fails or tracking fails
            print(f"[Detector Tracker] Warning in track(): {e}. Running predict without tracker...")
            results = self.model.predict(
                source=frames,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                verbose=False,
            )

        batch_outputs: List[List[TrackedVehicle]] = []

        for res in results:
            frame_vehicles: List[TrackedVehicle] = []
            boxes = getattr(res, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy_arr = boxes.xyxy.cpu().numpy()
                conf_arr = boxes.conf.cpu().numpy()
                cls_arr = boxes.cls.cpu().numpy().astype(int)

                # Track IDs if tracker succeeded, otherwise fallback index
                if boxes.id is not None:
                    id_arr = boxes.id.cpu().numpy().astype(int)
                else:
                    id_arr = np.arange(len(boxes), dtype=int)

                for i in range(len(boxes)):
                    c_id = int(cls_arr[i])
                    c_name = CLASS_NAMES.get(c_id, f"class_{c_id}")
                    v = TrackedVehicle(
                        track_id=int(id_arr[i]),
                        class_id=c_id,
                        class_name=c_name,
                        bbox=xyxy_arr[i].tolist(),
                        confidence=float(conf_arr[i]),
                        timestamp=timestamp,
                    )
                    frame_vehicles.append(v)

            batch_outputs.append(frame_vehicles)

        return batch_outputs

    def update_velocities(
        self,
        vehicles: List[TrackedVehicle],
        homography_func,
    ) -> None:
        """
        Projects vehicles to the ground plane and updates their estimated ground velocity.

        Args:
            vehicles: List of TrackedVehicle objects from the approach camera.
            homography_func: Callable (u, v) -> (X_m, Y_m).
        """
        now = time.time()
        active_ids = set()

        for v in vehicles:
            active_ids.add(v.track_id)
            gx, gy = homography_func(v.bottom_center[0], v.bottom_center[1])
            v.ground_pos = (gx, gy)

            if v.track_id not in self.track_trajectories:
                self.track_trajectories[v.track_id] = []

            traj = self.track_trajectories[v.track_id]
            traj.append((v.timestamp, gx, gy))

            if len(traj) > self.max_history_len:
                traj.pop(0)

            # Calculate speed over the last 3-5 frames (>= 0.1s delta)
            if len(traj) >= 3:
                t_old, x_old, y_old = traj[0]
                dt = v.timestamp - t_old
                if dt > 0.05:
                    dist_m = float(np.hypot(gx - x_old, gy - y_old))
                    speed = dist_m / dt
                    # Clamp unrealistic tracking jump noise
                    v.velocity_mps = min(35.0, max(0.0, speed))
                else:
                    v.velocity_mps = 0.0
            else:
                v.velocity_mps = 0.0

        # Prune stale trajectories older than 5 seconds
        for tid in list(self.track_trajectories.keys()):
            if tid not in active_ids:
                if self.track_trajectories[tid] and (now - self.track_trajectories[tid][-1][0] > 5.0):
                    del self.track_trajectories[tid]
