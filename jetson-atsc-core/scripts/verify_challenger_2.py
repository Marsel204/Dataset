#!/usr/bin/env python3
"""
Adversarial Empirical Verification Runner for Challenger 2.
Directly audits:
1. All 5 .jsonl telemetry traces: frame count vs video, sequential continuity, cycle records.
2. OpenCV video decoding: start, middle, and end frames across 5 benchmark videos and 5 comparative MP4s.
3. Homography conditioning: SVD condition numbers kappa(H), reprojection residuals epsilon, determinant.
"""

import json
import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TELEMETRY_BENCHMARK_PAIRS = [
    {
        "id": "scenario_1",
        "name": "Night Arterial / Glare",
        "trace": "data/telemetry_traces/trace_stress_night_arterial.jsonl",
        "video": "data/sample_videos/stress_test/night_glare.mp4",
        "config": "configs/stress_night_arterial.json",
    },
    {
        "id": "scenario_2",
        "name": "Rain & Wet Road",
        "trace": "data/telemetry_traces/trace_stress_rain_wet_road.jsonl",
        "video": "data/sample_videos/stress_test/rain_wet.mp4",
        "config": "configs/stress_rain_wet_road.json",
    },
    {
        "id": "scenario_3",
        "name": "Congestion Gridlock",
        "trace": "data/telemetry_traces/trace_stress_congestion_gridlock.jsonl",
        "video": "data/sample_videos/stress_test/congestion_gridlock.mp4",
        "config": "configs/stress_congestion_gridlock.json",
    },
    {
        "id": "scenario_4",
        "name": "Bandung Pasteur ATCS",
        "trace": "data/telemetry_traces/trace_stress_indonesia_bandung_pasteur.jsonl",
        "video": "data/sample_videos/multi_angle_test/video_bandung_pasteur_h264.mp4",
        "config": "configs/stress_indonesia_bandung_pasteur.json",
    },
    {
        "id": "scenario_5",
        "name": "Bekasi ATCS",
        "trace": "data/telemetry_traces/trace_stress_indonesia_bekasi.jsonl",
        "video": "data/sample_videos/multi_angle_test/video_bekasi_atcs.mp4",
        "config": "configs/stress_indonesia_bekasi.json",
    },
]

COMPARATIVE_VIDEOS = [
    ("Scenario 1 Night Glare", "data/comparative_media/compare_stress_night_arterial.mp4"),
    ("Scenario 2 Rain Wet", "data/comparative_media/compare_stress_rain_wet_road.mp4"),
    ("Scenario 3 Congestion", "data/comparative_media/compare_stress_congestion_gridlock.mp4"),
    ("Scenario 4 Bandung Pasteur", "data/comparative_media/compare_stress_indonesia_bandung_pasteur.mp4"),
    ("Scenario 5 Bekasi ATCS", "data/comparative_media/compare_stress_indonesia_bekasi.mp4"),
]


def audit_telemetry():
    print("=" * 80)
    print("1. TELEMETRY TRACES AUDIT (Exact Frame Counts, Zero Gaps, Cycle Records)")
    print("=" * 80)
    results = []
    all_passed = True

    for item in TELEMETRY_BENCHMARK_PAIRS:
        trace_path = os.path.join(PROJECT_ROOT, item["trace"])
        video_path = os.path.join(PROJECT_ROOT, item["video"])

        # Read trace records
        with open(trace_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        trace_count = len(records)

        # Video frames
        cap = cv2.VideoCapture(video_path)
        video_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Check frame_idx continuity
        gaps = []
        for i, rec in enumerate(records):
            if rec.get("frame_idx") != i:
                gaps.append((i, rec.get("frame_idx")))

        # Check complete cycle records
        cycle_records = [r for r in records if "decision" in r]
        cycles_info = []
        for r in cycle_records:
            d = r["decision"]
            cycles_info.append({
                "cycle_id": d.get("cycle_id"),
                "frame_idx": r.get("frame_idx"),
                "timestamp": d.get("timestamp"),
                "t_anfis": d.get("t_anfis"),
                "inputs": d.get("inputs"),
            })

        status = (trace_count == video_count) and (len(gaps) == 0) and (len(cycle_records) >= 1)
        if not status:
            all_passed = False

        res = {
            "name": item["name"],
            "trace_file": os.path.basename(item["trace"]),
            "video_file": os.path.basename(item["video"]),
            "trace_frames": trace_count,
            "video_frames": video_count,
            "count_match": trace_count == video_count,
            "zero_gaps": len(gaps) == 0,
            "gap_count": len(gaps),
            "cycle_count": len(cycle_records),
            "cycles": cycles_info,
            "passed": status,
        }
        results.append(res)
        print(f"[{'PASS' if status else 'FAIL'}] {item['name']}:")
        print(f"  Trace frames: {trace_count} | Video frames: {video_count} (Match: {res['count_match']})")
        print(f"  Frame continuity: {'ZERO GAPS' if len(gaps) == 0 else f'GAPS FOUND: {len(gaps)}'}")
        print(f"  Actuation cycles logged: {len(cycle_records)}")
        for c in cycles_info:
            print(f"    - Cycle {c['cycle_id']} @ Frame {c['frame_idx']} (t={c['timestamp']:.2f}s): "
                  f"t_ANFIS={c['t_anfis']}s, Inputs [Vw, Q, L]={c['inputs']}")

    return all_passed, results


def audit_video_decodability():
    print("\n" + "=" * 80)
    print("2. OPENCV VIDEO DECODING AUDIT (Start, Middle, and End Frames)")
    print("=" * 80)
    all_passed = True
    results = []

    # Benchmark videos
    print("\n--- 2.1 Benchmark Source Videos ---")
    for item in TELEMETRY_BENCHMARK_PAIRS:
        video_path = os.path.join(PROJECT_ROOT, item["video"])
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"Cannot open {video_path}"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        indices = [0, total_frames // 2, total_frames - 1]
        frame_stats = {}
        video_ok = True

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                video_ok = False
                frame_stats[idx] = {"ret": False, "shape": None, "mean": 0.0}
            else:
                frame_stats[idx] = {
                    "ret": True,
                    "shape": frame.shape,
                    "mean": float(np.mean(frame)),
                    "min": int(np.min(frame)),
                    "max": int(np.max(frame)),
                }

        cap.release()
        if not video_ok:
            all_passed = False

        res = {
            "type": "benchmark",
            "name": item["name"],
            "file": os.path.basename(item["video"]),
            "resolution": f"{w}x{h}",
            "fps": fps,
            "total_frames": total_frames,
            "frame_stats": frame_stats,
            "passed": video_ok,
        }
        results.append(res)
        print(f"[{'PASS' if video_ok else 'FAIL'}] {item['name']} ({res['file']}): {w}x{h} @ {fps:.1f} FPS, {total_frames} frames")
        for idx in indices:
            st = frame_stats[idx]
            tag = "start" if idx == 0 else ("mid" if idx == total_frames // 2 else "end")
            print(f"    - Frame {idx:4d} ({tag:5s}): ret={st['ret']}, shape={st['shape']}, mean_intensity={st['mean']:.2f}")

    # Comparative media videos
    print("\n--- 2.2 Comparative Split-Screen MP4 Videos ---")
    for name, rel_path in COMPARATIVE_VIDEOS:
        video_path = os.path.join(PROJECT_ROOT, rel_path)
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"Cannot open {video_path}"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        indices = [0, total_frames // 2, total_frames - 1]
        frame_stats = {}
        video_ok = True

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                video_ok = False
                frame_stats[idx] = {"ret": False, "shape": None, "mean": 0.0}
            else:
                frame_stats[idx] = {
                    "ret": True,
                    "shape": frame.shape,
                    "mean": float(np.mean(frame)),
                    "min": int(np.min(frame)),
                    "max": int(np.max(frame)),
                }

        cap.release()
        if not video_ok:
            all_passed = False

        res = {
            "type": "comparative",
            "name": name,
            "file": os.path.basename(rel_path),
            "resolution": f"{w}x{h}",
            "fps": fps,
            "total_frames": total_frames,
            "frame_stats": frame_stats,
            "passed": video_ok,
        }
        results.append(res)
        print(f"[{'PASS' if video_ok else 'FAIL'}] {name} ({res['file']}): {w}x{h} @ {fps:.1f} FPS, {total_frames} frames")
        for idx in indices:
            st = frame_stats[idx]
            tag = "start" if idx == 0 else ("mid" if idx == total_frames // 2 else "end")
            print(f"    - Frame {idx:4d} ({tag:5s}): ret={st['ret']}, shape={st['shape']}, mean_intensity={st['mean']:.2f}")

    return all_passed, results


def audit_homography():
    print("\n" + "=" * 80)
    print("3. HOMOGRAPHY CONDITIONING & REPROJECTION AUDIT (kappa(H) < 10^6, epsilon < 0.05m)")
    print("=" * 80)
    all_passed = True
    results = []

    for item in TELEMETRY_BENCHMARK_PAIRS:
        config_path = os.path.join(PROJECT_ROOT, item["config"])
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        src = np.array(cfg["homography"]["src_points_pixel"], dtype=np.float64)
        dst = np.array(cfg["homography"]["dst_points_ground"], dtype=np.float64)
        dims = cfg["homography"]["ground_dimensions_m"]

        H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32)).astype(np.float64)
        H_inv = cv2.getPerspectiveTransform(dst.astype(np.float32), src.astype(np.float32)).astype(np.float64)

        # Condition numbers
        kappa_h = float(np.linalg.cond(H))
        kappa_h_inv = float(np.linalg.cond(H_inv))

        # Forward reprojection error
        src_homo = np.hstack([src, np.ones((4, 1), dtype=np.float64)])
        dst_proj_homo = (H @ src_homo.T).T
        dst_proj = dst_proj_homo[:, :2] / dst_proj_homo[:, 2:3]
        fwd_errors_m = np.linalg.norm(dst_proj - dst, axis=1)
        max_fwd_err = float(np.max(fwd_errors_m))
        mean_fwd_err = float(np.mean(fwd_errors_m))

        # Inverse reprojection error
        dst_homo = np.hstack([dst, np.ones((4, 1), dtype=np.float64)])
        src_proj_homo = (H_inv @ dst_homo.T).T
        src_proj = src_proj_homo[:, :2] / src_proj_homo[:, 2:3]
        inv_errors_px = np.linalg.norm(src_proj - src, axis=1)
        max_inv_err = float(np.max(inv_errors_px))

        det_h = float(np.linalg.det(H))

        passed = (kappa_h < 1e6) and (kappa_h_inv < 1e6) and (max_fwd_err < 0.05) and (max_inv_err < 1.0)
        if not passed:
            all_passed = False

        res = {
            "name": item["name"],
            "config": os.path.basename(item["config"]),
            "corridor_w_l": f"{dims['width']}m x {dims['length']}m",
            "det_H": det_h,
            "kappa_H": kappa_h,
            "kappa_H_inv": kappa_h_inv,
            "max_epsilon_m": max_fwd_err,
            "mean_epsilon_m": mean_fwd_err,
            "max_epsilon_inv_px": max_inv_err,
            "passed": passed,
            "H_matrix": H.tolist(),
        }
        results.append(res)
        print(f"[{'PASS' if passed else 'FAIL'}] {item['name']} ({res['config']}):")
        print(f"  Corridor: {res['corridor_w_l']} | det(H): {det_h:.4e}")
        print(f"  Condition Number kappa(H): {kappa_h:.4e} (Threshold: < 1.0e+06) -> {'OK' if kappa_h < 1e6 else 'FAIL'}")
        print(f"  Condition Number kappa(H^-1): {kappa_h_inv:.4e} (Threshold: < 1.0e+06) -> {'OK' if kappa_h_inv < 1e6 else 'FAIL'}")
        print(f"  Max Forward Reprojection Error epsilon: {max_fwd_err:.6e} m (Threshold: < 0.05 m) -> {'OK' if max_fwd_err < 0.05 else 'FAIL'}")
        print(f"  Mean Forward Reprojection Error: {mean_fwd_err:.6e} m")
        print(f"  Max Inverse Reprojection Error: {max_inv_err:.6e} px (Threshold: < 1.0 px) -> {'OK' if max_inv_err < 1.0 else 'FAIL'}")

    return all_passed, results


if __name__ == "__main__":
    t_pass, t_res = audit_telemetry()
    v_pass, v_res = audit_video_decodability()
    h_pass, h_res = audit_homography()

    overall = t_pass and v_pass and h_pass
    print("\n" + "=" * 80)
    print(f"OVERALL ADVERSARIAL VERDICT: {'APPROVE' if overall else 'REQUEST_CHANGES'}")
    print("=" * 80)
    sys.exit(0 if overall else 1)
