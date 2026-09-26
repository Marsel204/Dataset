# E2E Test Infra: Jetson ATSC Stress & Indonesian ATCS Deployment Testing

## Test Philosophy
- Requirement-driven, opaque-box, and end-to-end verification.
- Validates real video inputs, lens unwarping, metric IPM homography, telemetry outputs, ANFIS safety bounds, and side-by-side media.

## Feature Inventory & Test Mapping
| # | Feature | Source | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|---------|--------|:------:|:------:|:------:|:------:|
| F1 | Multi-Condition Video Acquisition | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| F2 | AV1 Transcoding & Ingestion | USER_UPDATE §1 | 5 | 5 | ✓ | ✓ |
| F3 | Camera Calibration & IPM Validation | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| F4 | Indonesian PKJI 2014 Calibration | USER_UPDATE §2 | 5 | 5 | ✓ | ✓ |
| F5 | Replay Execution Stability | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| F6 | Telemetry Export & Integrity | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| F7 | False Stationary Queue Suppression | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| F8 | ANFIS Green Allocation Bounds | ORIGINAL_REQUEST §ANFIS | 5 | 5 | ✓ | ✓ |
| F9 | Side-by-Side Comparison Media | ORIGINAL_REQUEST §R4 | 5 | 5 | ✓ | ✓ |
| F10| Unit & Integration Test Regression | ORIGINAL_REQUEST §Tests | 5 | 5 | ✓ | ✓ |

## Test Architecture
- **E2E Test Runner**: Python-based automated suite executing `.venv/bin/pytest tests/` and verification scripts.
- **Verification Scripts**:
  - `verify_videos.py`: Checks presence, resolution (1920x1080), codecs, duration, frame counts.
  - `verify_calibration.py`: Validates 7 physical criteria (convex quad, $\kappa(H) < 10^6$, $\epsilon < 0.05\text{m}$, non-negative coordinates, metric bounds).
  - `verify_telemetry.py`: Checks `.jsonl` trace continuity, zero missing frames, cycle decision presence, and metric ranges.
  - `verify_anfis.py`: Checks $t_{\text{ANFIS}} \in [10.0\text{s}, 120.0\text{s}]$, dynamic scaling $>10.0\text{s}$ under demand.
  - `verify_media.py`: Checks side-by-side MP4 generation, H.264 codec, resolution, snapshot frames.

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Target Video |
|---|----------|--------------------|--------------|
| 1 | Night Low-Light & Glare | F1, F3, F5, F6, F7, F8, F9 | `data/sample_videos/stress_test/night_glare.mp4` |
| 2 | Rain & Wet Pavement Reflections | F1, F3, F5, F6, F7, F8, F9 | `data/sample_videos/stress_test/rain_wet.mp4` |
| 3 | Heavy Congestion & Gridlock | F1, F3, F5, F6, F7, F8, F9 | `data/sample_videos/stress_test/congestion_gridlock.mp4` |
| 4 | Bandung Pasteur Mixed Swarm (Indonesian) | F2, F4, F5, F6, F8, F9 | `data/sample_videos/multi_angle_test/video_bandung_pasteur_h264.mp4` |
| 5 | Bekasi ATCS Signalized Intersection (Indonesian) | F4, F5, F6, F8, F9 | `data/sample_videos/multi_angle_test/video_bekasi_atcs.mp4` |

## Acceptance Criteria
- 100% completion with zero unhandled exceptions across all 5 test sequences in `scripts/replay_debug.py`.
- Complete cycle records and zero missing frames in all exported `.jsonl` telemetry files.
- Sugeno ANFIS green allocations strictly adhere to safe bounds $[10.0\text{ s}, 120.0\text{ s}]$.
- Arriving flow and standing queues dynamically scale green times above baseline ($> 10.0\text{ s}$) when demand is detected.
- Wet road reflections and headlight glare do not trigger false stationary queues.
- Synchronized side-by-side comparison MP4 videos (H.264 encoded) and snapshot frames produced for each scenario.
- All 19 unit & integration tests pass with zero regressions.
