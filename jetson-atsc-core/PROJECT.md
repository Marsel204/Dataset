# Project: Jetson ATSC Edge Runtime Real-World Stress-Testing & Indonesian ATCS Deployment Benchmark

## Architecture
- **Perception Pipeline**:
  - Ingestion: OpenCV VideoCapture (H.264 / VP9 / standard codecs)
  - Distortion Correction: `LensRectifier` via `configs/rectilinear_calib.npz` (or camera-specific maps)
  - Detection: YOLO11s (`models/yolo11s.pt` / `models/yolo11s_fp16.engine`) detecting motorcycles, cars, trucks, buses
  - Tracking: `DetectorTracker` (DeepOCSORT / Kalman filter tracking with trajectory history)
- **Analytics & Geometry**:
  - Homography: `HomographyEngine` computing $3\times 3$ perspective transformation matrix $H$ mapping pixel $(u, v) \to$ metric ground plane $(X, Y\text{ in meters})$
  - Metric Corridor Filtering: Polygon point-in-corridor test isolating approach lanes (standard width $3.5\text{ m}$, reach up to $75\text{ m}$)
  - Traffic Metrics: `TrafficMetricsExtractor` extracting:
    - Weighted Flow $V_w$ in PCU (PKJI 2014: 0.4 MC, 1.0 LV, 1.6 HV)
    - Queue Length $Q$ in meters (stopped vehicles $v \le 1.0\text{ m/s}$)
    - Lane Occupancy $L$ in % ($A_{\text{vehicles}} / A_{\text{road}} \times 100$)
- **Signal Control & Actuation**:
  - Sugeno ANFIS Controller: `ANFISInferenceEngine` running 126-parameter 27-rule model (`models/anfis_sugeno.onnx`)
  - Decision Bounds: Clamped to $[10.0\text{ s}, 120.0\text{ s}]$
  - Actuation Triggers: Optical Yellow Detector (`PhaseDetector`) or Periodic Trigger Fallback (`--trigger_interval_sec`)
- **Telemetry & Media**:
  - Structured `.jsonl` trace exporter recording cycle-by-cycle metrics and actuation decisions
  - Side-by-side synchronized split-screen comparison video generator (`raw_feed` vs `AI_annotated_boxes`) encoded via H.264
  - Comprehensive pre-deployment certification report (`walkthrough.md`)

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | R1.1 Night Video Acquisition | Acquire 1080p night traffic video with headlight glare & high dynamic range | M1 | ORIGINAL_REQUEST §R1 |
| 2 | R1.2 Rain Video Acquisition | Acquire 1080p adverse weather video with wet road reflections | M1 | ORIGINAL_REQUEST §R1 |
| 3 | R1.3 Congestion Video Acquisition | Acquire 1080p dense urban congestion video with queue spillback | M1 | ORIGINAL_REQUEST §R1 |
| 4 | R1.4 Indonesian Feed Ingestion & AV1 Transcoding | Transcode `video_bandung_pasteur.mp4` to H.264; validate `video_bekasi_atcs.mp4` | M1 | USER_UPDATE §1 |
| 5 | R2.1 Calibration Schema & Math Validation | Automated physical validator: convex quad, $\kappa(H) < 10^6$, $\epsilon < 0.05\text{m}$, non-negative | M2 | ORIGINAL_REQUEST §R2 |
| 6 | R2.2 Stress Calibration Profiles | Calibration JSONs for Night, Rain, and Congestion scenarios | M2 | ORIGINAL_REQUEST §R2 |
| 7 | R2.3 Indonesian Calibration Profiles | Calibration JSONs for Bandung Pasteur & Bekasi with PKJI 2014 PCE (0.4, 1.0, 1.6) | M2 | USER_UPDATE §2 |
| 8 | R3.1 Telemetry Serialization Fix | Patch `replay_debug.py:215` so decisions are logged in fallback cycle mode | M3 | Survey Explorer 1 |
| 9 | R3.2 False Stationary Queue Filter | Add temporal persistence gate ($N \ge 5$) to prevent glare/reflection false queues | M3 | Survey Explorer 3 |
| 10 | R3.3 Systematic Replay Execution | Execute `replay_debug.py` across all 5 scenarios with 0 unhandled exceptions | M3 | ORIGINAL_REQUEST §R3 |
| 11 | R3.4 Telemetry Export Validation | Validate `.jsonl` traces have complete cycle records and zero missing frames | M3 | ORIGINAL_REQUEST §R3 |
| 12 | R3.5 Real-time Performance Audit | Profile FPS ($\ge 30$) and latency ($\le 33\text{ ms}$) across benchmark runs | M3 | ORIGINAL_REQUEST §R3 |
| 13 | R4.1 ANFIS Actuation Verification | Verify safe bounds $[10.0\text{s}, 120.0\text{s}]$ and dynamic demand scaling ($>10.0\text{s}$) | M4 | ORIGINAL_REQUEST §ANFIS |
| 14 | R4.2 Side-by-Side Video Generation | Render synchronized horizontal split-screen MP4s (H.264) for all scenarios | M4 | ORIGINAL_REQUEST §R4 |
| 15 | R4.3 Snapshot Frame Extraction | Extract high-res snapshot frames highlighting raw vs annotated detection | M4 | ORIGINAL_REQUEST §R4 |
| 16 | R4.4 Test Suite Regression Verification | Run `.venv/bin/pytest tests/` verifying 19/19 tests pass | M4 | ORIGINAL_REQUEST §Tests |
| 17 | R4.5 Pre-Deployment Walkthrough Audit | Compile complete `walkthrough.md` with telemetry, media, and readiness analysis | M4 | ORIGINAL_REQUEST §R4 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Multi-Condition Dataset Acquisition & Transcoding | Download & standardize Night, Rain, Congestion videos in `data/sample_videos/stress_test/`; transcode Bandung AV1 to H.264; verify Bekasi VP9 | None | DONE (All 5 streams verified 1080p H.264, 19/19 tests pass) |
| M2 | Camera Calibration & IPM Profile Generation | Generate calibration JSONs for all 5 scenarios with physical validation & PKJI 2014 PCE | M1 | DONE (All 5 profiles + 2 aliases physically validated, 56 new tests pass, 75/75 total tests pass) |
| M3 | Pipeline Enhancements & Stress Replay Execution | Fix `replay_debug.py` telemetry bug, add false queue filter, execute replay benchmarks, export `.jsonl` traces | M2 | DONE (All 5 benchmarks executed across 4,357 frames, 0 exceptions, 5 .jsonl traces, 98/98 tests pass) |
| M4 | ANFIS Validation, Side-by-Side Media & Walkthrough | Validate ANFIS actuation, generate H.264 side-by-side MP4s & snapshots, verify 19 tests, compile `walkthrough.md` | M3 | DONE (Side-by-side H.264 videos & snapshots generated, walkthrough.md compiled, 119/119 tests pass) |

## Interface Contracts
### Video Input ↔ Replay Pipeline
- Format: MP4 container, H.264 / VP9 / standard OpenCV-decodable codec
- Resolution: $1920\times 1080$, Framerate: $10 - 60\text{ FPS}$

### Calibration Config ↔ Pipeline
- JSON schema:
  - `system.model_path`: YOLO model path
  - `system.calibration_path`: Lens unwarp NPZ path (`configs/rectilinear_calib.npz`)
  - `homography.src_points_pixel`: 4 image pixel coordinates $[u, v]$ forming convex quad
  - `homography.dst_points_ground`: 4 ground metric coordinates $[X, Y]$ in meters
  - `lane_metrics.corridor_polygon`: Polygon bounding inbound approach corridor
  - `lane_metrics.pcu_weights`: `{"motorcycle": 0.4, "car": 1.0, "truck": 1.6, "bus": 1.6}`
  - `traffic_light_monitor.active_ratio_threshold`: 0.99 for non-optical fallback

### Replay Runner ↔ Telemetry Sink
- Output: `.jsonl` line-delimited JSON
- Per-frame: `{"frame_idx": int, "timestamp": float, "processing_time_ms": float, "fps": float, "metrics": {"V_w": float, "Q": float, "L": float}}`
- Per-cycle: `record["decision"] = {"t_anfis": float, "inputs": [V_w, Q, L], "timestamp": float}`

## Code Layout
- `src/perception/`: `detector_tracker.py`, `lens_rectifier.py`, `phase_detector.py`
- `src/analytics/`: `homography_engine.py`, `traffic_metrics.py`
- `src/control/`: `anfis_inference.py`, `decision_engine.py`
- `src/telemetry/`: `signal_recorder.py`, `system_telemetry.py`
- `scripts/`: `replay_debug.py`, `export_models.py`
- `configs/`: calibration JSON profiles and `rectilinear_calib.npz`
- `data/sample_videos/stress_test/`: stress-test video sequences
- `tests/`: 19 unit & integration tests
