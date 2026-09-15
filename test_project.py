import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from torch2rtl.models import DemoMLP, TinyCNN
from torch2rtl.analyzer import analyze_model
from torch2rtl.quant import quantize_tensor
from torch2rtl.planner import make_plan

def test_mlp_analysis():
    _, layers = analyze_model(DemoMLP(), torch.randn(1,16))
    assert any(x.op == "Linear" for x in layers)
    assert sum(x.params for x in layers) > 0

def test_cnn_analysis():
    _, layers = analyze_model(TinyCNN(), torch.randn(1,1,28,28))
    assert any(x.op == "Conv2d" for x in layers)
    assert sum(x.macs for x in layers) > 0

def test_quantization_range():
    x = torch.tensor([-1000.0, -0.3, 0.3, 1000.0])
    q = quantize_tensor(x, 8, 4)
    assert q.max() <= 127/16
    assert q.min() >= -128/16

def test_parallel_plan():
    _, layers = analyze_model(DemoMLP(), torch.randn(1,16))
    p1 = make_plan(layers, parallelism=1)
    p4 = make_plan(layers, parallelism=4)
    assert p4.estimated_cycles <= p1.estimated_cycles
