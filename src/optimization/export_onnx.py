"""Export mood CNN and keypoint models to ONNX, then apply INT8 dynamic
quantization via onnxruntime. Produces (per model): a .onnx fp32 file and
a .onnx int8 file in artifacts/onnx/.
"""
import sys
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import load_config, resolve_path  # noqa: E402
from src.keypoints.model import HeatmapKeypointModel  # noqa: E402
from src.mood_cnn.model import MoodCNN  # noqa: E402


def export_and_quantize(model: torch.nn.Module, dummy_input: torch.Tensor, out_dir: Path, name: str, output_names):
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / f"{name}_fp32.onnx"
    int8_path = out_dir / f"{name}_int8.onnx"

    model.eval()
    # Fixed batch size (1) — this export is for benchmarking single-image
    # inference latency, not serving variable batch sizes. Forcing an
    # opset downgrade (e.g. requesting 17 when the exporter defaults to 18)
    # triggered a broken graph conversion in torch 2.13's dynamo-based
    # exporter (shape-inference errors downstream in quantization) —
    # letting it use its own default opset avoids that. external_data=False
    # keeps weights embedded in the single .onnx file (the exporter's
    # default splits them into a companion .onnx.data file, which both
    # breaks a simple single-file size comparison and confused the
    # quantizer's shape-inference step).
    torch.onnx.export(
        model, dummy_input, str(fp32_path),
        input_names=["input"], output_names=output_names,
        external_data=False,
    )
    print(f"exported {fp32_path} ({fp32_path.stat().st_size / 1e6:.2f} MB)")

    # torch 2.13's dynamo-based exporter produces graphs that pass onnx's
    # own checker and run correctly, but trip up onnxruntime's quantizer's
    # internal (stricter) shape-inference pass. quant_pre_process is
    # onnxruntime's own recommended fix — a more tolerant shape-inference
    # + optimization pass specifically meant to run before quantization.
    preprocessed_path = out_dir / f"{name}_fp32_preprocessed.onnx"
    quant_pre_process(str(fp32_path), str(preprocessed_path))

    quantize_dynamic(
        model_input=str(preprocessed_path), model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    print(f"quantized {int8_path} ({int8_path.stat().st_size / 1e6:.2f} MB)")
    return fp32_path, int8_path


def main():
    cfg = load_config()
    device = torch.device("cpu")  # export on CPU; ONNX/quantization benchmarking targets CPU inference
    onnx_dir = resolve_path(cfg["paths"]["artifacts"]) / "onnx"

    mcfg = cfg["mood_cnn"]
    mood_model = MoodCNN(num_classes=len(cfg["mood_classes"]), embedding_dim=mcfg["embedding_dim"], pretrained=False)
    mood_model.load_state_dict(torch.load(resolve_path(cfg["paths"]["mood_cnn_model"]), map_location=device))
    dummy = torch.randn(1, 3, mcfg["input_size"], mcfg["input_size"])
    export_and_quantize(mood_model, dummy, onnx_dir, "mood_cnn", ["logits"])

    kcfg = cfg["keypoints"]
    kp_model = HeatmapKeypointModel(num_keypoints=kcfg["num_keypoints"], pretrained=False)
    kp_model.load_state_dict(torch.load(resolve_path(cfg["paths"]["keypoint_model"]), map_location=device))
    dummy_kp = torch.randn(1, 3, kcfg["input_size"], kcfg["input_size"])
    export_and_quantize(kp_model, dummy_kp, onnx_dir, "keypoint_model", ["heatmaps"])


if __name__ == "__main__":
    main()
