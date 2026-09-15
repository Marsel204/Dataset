"""
Model Export Utility for Jetson ATSC Production Runtime.

Exports:
1. PyTorch 126-parameter Sugeno ANFIS to ONNX format (models/anfis_sugeno.onnx)
   with integrated MinMax normalization and green-time clamping bounds [10s, 120s].
2. Ultralytics YOLO11 model (Final.pt) to TensorRT FP16 engine (models/yolo11s_fp16.engine).
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn

# Add project roots
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class FuzzificationLayer(nn.Module):
    """Layer 1: Gaussian Membership Functions."""
    def __init__(self, centers: torch.Tensor, sigmas: torch.Tensor):
        super().__init__()
        self.centers = nn.Parameter(centers)
        self.sigmas = nn.Parameter(sigmas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 3)
        x_exp = x.unsqueeze(-1)  # (batch_size, 3, 1)
        c_exp = self.centers.unsqueeze(0)  # (1, 3, 3)
        s_exp = torch.clamp(self.sigmas.abs(), min=1e-6).unsqueeze(0)
        diff = (x_exp - c_exp) / s_exp
        return torch.exp(-0.5 * (diff ** 2))


class RuleFiringLayer(nn.Module):
    """Layer 2: Product T-norm across 27 rules."""
    def __init__(self, rule_indices: torch.Tensor):
        super().__init__()
        self.register_buffer("rule_indices", rule_indices)

    def forward(self, memberships: torch.Tensor) -> torch.Tensor:
        # memberships: (batch_size, 3, 3)
        w0 = memberships[:, 0, self.rule_indices[:, 0]]
        w1 = memberships[:, 1, self.rule_indices[:, 1]]
        w2 = memberships[:, 2, self.rule_indices[:, 2]]
        return w0 * w1 * w2


class NormalizationLayer(nn.Module):
    """Layer 3: Normalize rule firing strengths."""
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w / (torch.sum(w, dim=-1, keepdim=True) + self.eps)


class ConsequentLayer(nn.Module):
    """Layer 4: First-order Sugeno linear consequences."""
    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.consequent_weights = nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        ones = torch.ones(batch_size, 1, dtype=x.dtype, device=x.device)
        x_aug = torch.cat([x, ones], dim=-1)
        return torch.matmul(x_aug, self.consequent_weights.t())


class OutputLayer(nn.Module):
    """Layer 5: Weighted sum of consequences."""
    def forward(self, w_bar: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        return torch.sum(w_bar * f, dim=-1, keepdim=True)


class ExportableANFIS(nn.Module):
    """
    Complete 5-Layer Sugeno ANFIS Model with integrated feature MinMax normalization
    and green duration clamping bounds [10.0, 120.0] seconds.
    Total learnable parameters = 18 premise + 108 consequent = 126 parameters.
    """
    def __init__(self, checkpoint_dict: dict):
        super().__init__()
        sd = checkpoint_dict["model_state_dict"]
        s_min = torch.tensor(checkpoint_dict["scaler_min"], dtype=torch.float32)
        s_max = torch.tensor(checkpoint_dict["scaler_max"], dtype=torch.float32)
        diff = torch.clamp(s_max - s_min, min=1e-6)

        self.register_buffer("scaler_min", s_min)
        self.register_buffer("scaler_diff", diff)

        self.layer1_fuzz = FuzzificationLayer(
            sd["layer1_fuzzification.centers"],
            sd["layer1_fuzzification.sigmas"]
        )
        self.layer2_rules = RuleFiringLayer(sd["layer2_rules.rule_indices"])
        self.layer3_norm = NormalizationLayer()
        self.layer4_conseq = ConsequentLayer(sd["layer4_consequent.consequent_weights"])
        self.layer5_out = OutputLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, 3) representing [V_w, Q, L]
        # 1. Feature normalization
        scaled_x = torch.clamp((x - self.scaler_min) / self.scaler_diff, 0.0, 1.0)

        # 2. ANFIS 5-layer forward pass
        mems = self.layer1_fuzz(scaled_x)
        w = self.layer2_rules(mems)
        w_bar = self.layer3_norm(w)
        f = self.layer4_conseq(scaled_x)
        raw_output = self.layer5_out(w_bar, f)

        # 3. Clamping to valid ATSC green-time duration bounds [10.0s, 120.0s]
        return torch.clamp(raw_output, 10.0, 120.0)


def export_anfis_onnx(checkpoint_path: str, output_onnx_path: str) -> None:
    """Exports trained PyTorch ANFIS checkpoint to ONNX."""
    print(f"[ANFIS Export] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = ExportableANFIS(checkpoint)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[ANFIS Export] Total model parameters: {total_params} (Expected: 126)")

    os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)
    dummy_input = torch.tensor([[25.0, 35.0, 20.0]], dtype=torch.float32)

    with torch.no_grad():
        baseline_pred = model(dummy_input)
        print(f"[ANFIS Export] Baseline PyTorch prediction for [25, 35, 20]: {baseline_pred.item():.4f}s")

    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        input_names=["traffic_metrics"],
        output_names=["green_duration"],
        dynamic_axes={
            "traffic_metrics": {0: "batch_size"},
            "green_duration": {0: "batch_size"}
        },
        opset_version=14,
        dynamo=False
    )
    print(f"[ANFIS Export] Saved ONNX model successfully to: {output_onnx_path}")


def export_yolo_tensorrt(weights_path: str, engine_path: str, imgsz: int = 640, half: bool = True) -> None:
    """Exports Ultralytics YOLO model to TensorRT engine."""
    print(f"[YOLO Export] Loading YOLO weights from: {weights_path}")
    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        print(f"[YOLO Export] Exporting to TensorRT format (half={half}, imgsz={imgsz})...")
        exported_file = model.export(
            format="engine",
            half=half,
            imgsz=imgsz,
            device=0 if torch.cuda.is_available() else "cpu",
            batch=2  # Enable batch-2 inference for DUAL_CAM mode
        )
        print(f"[YOLO Export] TensorRT engine compiled: {exported_file}")
        if exported_file != engine_path and os.path.exists(exported_file):
            import shutil
            os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
            shutil.copyfile(exported_file, engine_path)
            print(f"[YOLO Export] Copied engine to: {engine_path}")
    except Exception as e:
        print(f"[YOLO Export] TensorRT export failed or not supported on this host: {e}")
        print("[YOLO Export] Runtime will safely use Final.pt directly.")


def main():
    parser = argparse.ArgumentParser(description="Export ATSC models to ONNX and TensorRT")
    parser.add_argument(
        "--anfis_checkpoint",
        type=str,
        default="/home/marsel/Projects/ANFISProductionSystem/run_s1000_skewed/anfis_best_model.pt",
        help="Path to trained PyTorch ANFIS checkpoint"
    )
    parser.add_argument(
        "--anfis_output",
        type=str,
        default="/home/marsel/Projects/ANFISProductionSystem/jetson-atsc-core/models/anfis_sugeno.onnx",
        help="Path to output ANFIS ONNX file"
    )
    parser.add_argument(
        "--yolo_weights",
        type=str,
        default="/home/marsel/Projects/ANFISProductionSystem/Final.pt",
        help="Path to YOLO11 weights (.pt)"
    )
    parser.add_argument(
        "--yolo_engine",
        type=str,
        default="/home/marsel/Projects/ANFISProductionSystem/jetson-atsc-core/models/yolo11s_fp16.engine",
        help="Path to output TensorRT engine file"
    )
    parser.add_argument(
        "--skip_trt",
        action="store_true",
        help="Skip TensorRT compilation (useful when building on host without JetPack)"
    )

    args = parser.parse_args()

    # 1. Export ANFIS to ONNX
    export_anfis_onnx(args.anfis_checkpoint, args.anfis_output)

    # 2. Export YOLO to TensorRT (optional or if requested)
    if not args.skip_trt:
        export_yolo_tensorrt(args.yolo_weights, args.yolo_engine)


if __name__ == "__main__":
    main()
