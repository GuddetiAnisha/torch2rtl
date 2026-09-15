import argparse, json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))
from torch2rtl.models import build_model
from torch2rtl.analyzer import analyze_model
from torch2rtl.planner import make_plan
from torch2rtl.quant import compare_float_quantized
from torch2rtl.codegen import generate_rtl
from torch2rtl.report import write_outputs

def parse_shape(text):
    return tuple(int(x.strip()) for x in text.split(","))

def load_checkpoint(model, path):
    obj = torch.load(path, map_location="cpu")
    state = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
    model.load_state_dict(state)
    return model

def main():
    p = argparse.ArgumentParser(description="PyTorch-to-RTL software planning prototype")
    p.add_argument("--model", choices=["demo_mlp", "tiny_cnn"], default="demo_mlp")
    p.add_argument("--checkpoint")
    p.add_argument("--input-shape", default="1,16")
    p.add_argument("--word-bits", type=int, default=16)
    p.add_argument("--frac-bits", type=int, default=8)
    p.add_argument("--parallelism", type=int, default=4)
    p.add_argument("--out", default="generated")
    args = p.parse_args()

    torch.manual_seed(7)
    model = build_model(args.model)
    if args.checkpoint:
        model = load_checkpoint(model, args.checkpoint)
    sample = torch.randn(*parse_shape(args.input_shape))

    _, layers = analyze_model(model, sample)
    plan = make_plan(layers, args.word_bits, args.frac_bits, args.parallelism)
    verification = compare_float_quantized(model, sample, args.word_bits, args.frac_bits)
    hierarchy = generate_rtl(layers, args.out, args.word_bits)
    write_outputs(layers, plan, hierarchy, verification, args.out)
    print(json.dumps({
        "status": "ok", "layers": len(layers),
        "params": plan.total_params, "macs": plan.total_macs,
        "output": str(Path(args.out).resolve())
    }, indent=2))

if __name__ == "__main__":
    main()
