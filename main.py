import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp


class DemoMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, x):
        return self.net(x)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(16, 4)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model(name):
    if name == "demo_mlp":
        return DemoMLP()
    if name == "tiny_cnn":
        return TinyCNN()
    raise ValueError(f"Unsupported model: {name}")


def parse_shape(text):
    return tuple(int(x.strip()) for x in text.split(","))


def load_checkpoint(model, path):
    obj = torch.load(path, map_location="cpu")
    state = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
    model.load_state_dict(state)
    return model


def _shape_from_meta(node):
    meta = node.meta.get("tensor_meta")
    if meta is None:
        return None
    shape = getattr(meta, "shape", None)
    return list(shape) if shape is not None else None


def _estimate_module_macs(module, node):
    shape = _shape_from_meta(node)

    if isinstance(module, nn.Linear) and shape:
        batch = shape[0] if len(shape) > 1 else 1
        return int(batch * module.in_features * module.out_features)

    if isinstance(module, nn.Conv2d) and shape and len(shape) == 4:
        batch, out_channels, out_h, out_w = shape
        kernel_h, kernel_w = module.kernel_size
        per_output = (module.in_channels // module.groups) * kernel_h * kernel_w
        return int(batch * out_channels * out_h * out_w * per_output)

    return 0


def analyze_model(model, sample):
    model.eval()
    traced = symbolic_trace(model)

    try:
        ShapeProp(traced).propagate(sample)
    except Exception:
        pass

    modules = dict(traced.named_modules())
    nodes = []
    total_macs = 0

    for node in traced.graph.nodes:
        item = {
            "name": node.name,
            "op": node.op,
            "target": str(node.target),
            "inputs": [n.name for n in node.all_input_nodes],
            "shape": _shape_from_meta(node),
        }

        if node.op == "call_module" and str(node.target) in modules:
            module = modules[str(node.target)]
            item["module_type"] = module.__class__.__name__
            item["parameters"] = sum(p.numel() for p in module.parameters())
            item["estimated_macs"] = _estimate_module_macs(module, node)
            total_macs += item["estimated_macs"]

        nodes.append(item)

    with torch.no_grad():
        output = model(sample)

    return {
        "model": model.__class__.__name__,
        "input_shape": list(sample.shape),
        "output_shape": list(output.shape),
        "graph_node_count": len(nodes),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "estimated_macs": total_macs,
        "nodes": nodes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Software-only PyTorch computational graph analyser"
    )
    parser.add_argument(
        "--model",
        choices=["demo_mlp", "tiny_cnn"],
        default="demo_mlp",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--input-shape", default="1,16")
    parser.add_argument("--out", default="generated")
    args = parser.parse_args()

    torch.manual_seed(7)

    model = build_model(args.model)
    if args.checkpoint:
        model = load_checkpoint(model, args.checkpoint)

    sample = torch.randn(*parse_shape(args.input_shape))
    report = analyze_model(model, sample)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "model_analysis.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "model": report["model"],
                "graph_nodes": report["graph_node_count"],
                "parameters": report["trainable_parameters"],
                "estimated_macs": report["estimated_macs"],
                "output": str(output_path.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
