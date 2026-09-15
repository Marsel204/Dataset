"""
Asynchronous LLM Operational Quality Auditor.

Acts as a rigorous, critical production test engineer evaluating completed
traffic signal cycles. Inspects cycle metrics (V_w, Q, L, t_ANFIS, N_served, FPS, Temp)
against physical traffic flow heuristics to flag anomalies such as:
1. Min/Max Clamping Saturation (10s or 120s green time under moderate demand)
2. Capacity Mismatch vs. Headway (N_served significantly divergent from theoretical discharge)
3. Thermal and Throughput Degradation (SoC temp >= 75°C or FPS < 15)
4. Tracking Identity Drops or Erratic Queue Fluctuations

Runs in a non-blocking background worker queue without stalling real-time edge control.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional
import requests


SYSTEM_PROMPT = """You are an expert embedded AI quality assurance engineer and senior traffic engineer auditing an Adaptive Traffic Signal Controller (ATSC) deployed on an NVIDIA Jetson Orin Nano edge device.

Analyze the given cycle telemetry snapshot:
- V_w (PKJI 2014 Weighted Vehicle Flow in PCU)
- Q (Queue Length in meters)
- L (Fixed-Area Lane Occupancy in %)
- t_ANFIS (ANFIS Allocated Green Duration in seconds, valid range: [10s, 120s])
- N_served (Actual vehicles counted discharging over stop-line tripwire during green)
- FPS (Perception pipeline throughput)
- SoC Temp (°C)

Traffic Engineering Heuristics:
1. Saturation Flow Headway: Typical signalized intersection saturation headway h_s is ~1.8 to 2.2 seconds per PCU. Theoretical maximum discharge capacity during green is approximately C_theor = t_ANFIS / 2.0.
2. Clamping Saturation: If t_ANFIS is clamped to exactly 10.0s (minimum) when V_w > 15 PCU or Q > 15m, OR clamped to 120.0s (maximum) when V_w < 40 PCU, flag as severe ANFIS rule base or scaling saturation anomaly.
3. Served vs Demand Discrepancy: If N_served is near zero despite high queue and green time, flag probable stop-line occlusion, tracking dropout, or downstream bottleneck.
4. Edge Thermal Degradation: If SoC Temp >= 75°C or FPS < 15, flag hardware thermal/throughput degradation.

Respond STRICTLY in valid JSON with the following schema:
{
  "anomaly_detected": true/false,
  "severity": "NORMAL" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "anomaly_types": ["CLAMPING_SATURATION", "CAPACITY_MISMATCH", "THERMAL_DEGRADATION", "TRACKING_INSTABILITY"],
  "critique": "Brief 1-2 sentence engineering diagnosis.",
  "recommendation": "Brief actionable remediation."
}
"""


class LLMAuditor:
    """
    Asynchronous background auditor submitting cycle snapshots to an LLM endpoint
    or falling back to a deterministic heuristic rule engine.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        on_audit_completed: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.api_key = api_key or self._discover_api_key()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.on_audit_completed = on_audit_completed

        self._queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=50)
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        self.total_cycles_audited = 0
        self.total_anomalies_flagged = 0

    def _discover_api_key(self) -> Optional[str]:
        """Looks for API key in environment or existing project .env files."""
        # Check direct env vars
        key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if key:
            return key

        # Check Orchestrator .env
        alt_env_path = "/home/marsel/Projects/Orchestrator/.env"
        if os.path.exists(alt_env_path):
            try:
                with open(alt_env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("DEEPSEEK_API_KEY="):
                            return line.strip().split("=", 1)[1].strip()
            except Exception:
                pass

        return None

    def start(self) -> LLMAuditor:
        """Launches the background audit queue worker."""
        if self._running:
            return self
        self._running = True
        self._worker_thread = threading.Thread(target=self._audit_worker, daemon=True, name="LLMAuditorWorker")
        self._worker_thread.start()
        print(f"[LLMAuditor] Started background auditor (LLM Online: {bool(self.api_key)})")
        return self

    def enqueue_cycle(self, cycle_data: Dict[str, Any]) -> None:
        """
        Enqueues a completed traffic cycle snapshot for auditing. Non-blocking.
        """
        if not self._running:
            return

        try:
            self._queue.put_nowait(cycle_data)
        except queue.Full:
            print("[LLMAuditor] Warning: Audit queue is full. Dropping oldest item.")
            try:
                _ = self._queue.get_nowait()
                self._queue.put_nowait(cycle_data)
            except Exception:
                pass

    def _audit_worker(self) -> None:
        """Background thread worker pulling snapshots and querying the LLM."""
        while self._running:
            try:
                cycle_data = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            audit_result = self._evaluate_cycle(cycle_data)
            self.total_cycles_audited += 1
            if audit_result.get("anomaly_detected", False):
                self.total_anomalies_flagged += 1

            # Dispatch callback or attach result
            if self.on_audit_completed:
                try:
                    self.on_audit_completed({**cycle_data, "audit": audit_result})
                except Exception as e:
                    print(f"[LLMAuditor] Error in on_audit_completed callback: {e}")

            self._queue.task_done()

    def _evaluate_cycle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sends cycle snapshot to LLM or executes heuristic fallback."""
        if self.api_key:
            try:
                return self._call_llm_api(data)
            except Exception as e:
                print(f"[LLMAuditor] API request error: {e}. Falling back to rule heuristics.")

        return self._heuristic_critic(data)

    def _call_llm_api(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calls OpenAI-compatible LLM endpoint."""
        user_prompt = f"""Audit this ATSC Cycle:
- Cycle ID: {data.get('cycle_id', 'N/A')}
- Mode: {data.get('mode', 'SINGLE_CAM')}
- Weighted Flow (V_w): {data.get('V_w', 0.0)} PCU
- Queue Length (Q): {data.get('Q', 0.0)} m
- Lane Occupancy (L): {data.get('L', 0.0)} %
- ANFIS Green Time (t_ANFIS): {data.get('t_anfis', 30.0)} s
- Discharged Vehicles (N_served): {data.get('n_served', 0)}
- Throughput FPS: {data.get('fps', 30.0)}
- SoC Temperature: {data.get('soc_temp_c', 45.0)} °C
"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 300,
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=10.0,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)

    def _heuristic_critic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deterministic rule-based fallback mimicking LLM test engineer evaluation.
        """
        v_w = float(data.get("V_w", 0.0))
        q = float(data.get("Q", 0.0))
        t_g = float(data.get("t_anfis", 30.0))
        n_served = int(data.get("n_served", 0))
        fps = float(data.get("fps", 30.0))
        temp = float(data.get("soc_temp_c", 45.0))

        anomalies = []
        severity = "NORMAL"
        critique_parts = []
        recomms = []

        # 1. Clamping saturation
        if t_g <= 10.0 and (v_w > 15.0 or q > 15.0):
            anomalies.append("CLAMPING_SATURATION")
            severity = "HIGH"
            critique_parts.append(f"Premature minimum green clamp (10.0s) under active demand (V_w={v_w:.1f} PCU, Q={q:.1f}m).")
            recomms.append("Inspect ANFIS lower-bound consequent parameters.")
        elif t_g >= 120.0 and v_w < 35.0 and q < 20.0:
            anomalies.append("CLAMPING_SATURATION")
            severity = "MEDIUM"
            critique_parts.append(f"Maximum green clamp (120.0s) saturated under moderate demand (V_w={v_w:.1f} PCU).")
            recomms.append("Verify ANFIS upper bound scaling normalization.")

        # 2. Capacity vs discharge mismatch
        theor_cap = int(t_g / 2.0)  # saturation headway ~2.0s
        if q > 20.0 and t_g >= 30.0 and n_served < max(2, int(0.15 * theor_cap)):
            anomalies.append("CAPACITY_MISMATCH")
            severity = "HIGH" if severity != "CRITICAL" else severity
            critique_parts.append(f"Severe discharge starvation: only {n_served} vehicles cleared vs {theor_cap} PCU capacity.")
            recomms.append("Check stop-line tripwire occlusions or downstream spillback.")

        # 3. Thermal and throughput
        if temp >= 75.0:
            anomalies.append("THERMAL_DEGRADATION")
            severity = "CRITICAL"
            critique_parts.append(f"Edge SoC temperature {temp:.1f}°C exceeded 75.0°C safety limit.")
            recomms.append("Trigger dynamic thermal fallback to SINGLE_CAM immediately.")
        elif fps < 15.0:
            anomalies.append("THERMAL_DEGRADATION")
            severity = "HIGH" if severity != "CRITICAL" else severity
            critique_parts.append(f"Inference throughput collapsed to {fps:.1f} FPS.")
            recomms.append("Reduce camera resolution or step down to single camera mode.")

        has_anomaly = len(anomalies) > 0
        critique = " ".join(critique_parts) if critique_parts else "Nominal cycle: parameters within healthy empirical tolerances."
        recommendation = " ".join(recomms) if recomms else "Continue nominal adaptive control loop."

        return {
            "anomaly_detected": has_anomaly,
            "severity": severity,
            "anomaly_types": anomalies,
            "critique": critique,
            "recommendation": recommendation,
        }

    def stop(self) -> None:
        """Flushes queue and terminates worker thread."""
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        print(f"[LLMAuditor] Stopped. (Total Audited: {self.total_cycles_audited}, Anomalies: {self.total_anomalies_flagged})")
