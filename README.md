# Torch2RTL Planner (Software Prototype)

A Python-only research prototype inspired by AI-to-RTL toolchains. It **does not synthesize or deploy to FPGA hardware**. Instead, it focuses on the software/compiler side:

1. Load or construct a PyTorch model
2. Trace its computational graph with `torch.fx`
3. Extract supported layers and tensor shapes
4. Estimate MACs, parameters, memory and latency proxies
5. Choose configurable fixed-point and parallelism settings
6. Generate **SystemVerilog skeleton modules** (not production RTL)
7. Generate an RTL hierarchy manifest and synthesis-planning report
8. Validate the PyTorch model against a software fixed-point simulator

## Why this is intentionally different

The thesis description targets a full end-to-end synthesizable FPGA/ASIC flow. This project is deliberately adapted into a **software architecture exploration and RTL planning tool**. It demonstrates PyTorch graph analysis, compiler-style intermediate representation, quantization simulation, design-space exploration and code generation without requiring Vivado/Quartus or physical FPGA hardware.

## Supported model layers

- `nn.Linear`
- `nn.ReLU`
- `nn.Flatten`
- `nn.Conv2d`
- `nn.MaxPool2d`
- `nn.AdaptiveAvgPool2d`
- common FX operations such as reshape/flatten are recorded as generic operations

## Quick start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
python examples/train_demo.py
python main.py --checkpoint outputs/demo_mlp.pth --model demo_mlp --input-shape 1,16
```

CNN example:

```bash
python main.py --model tiny_cnn --input-shape 1,1,28,28 --word-bits 16 --frac-bits 8 --parallelism 4
```

Generated files are placed under `generated/`:
- `model_ir.json`
- `planning_report.md`
- `rtl_hierarchy.json`
- `rtl/*.sv`
- `build_manifest.json`
- `verification.json`

## Important limitation

Generated SystemVerilog is an **interface/architecture skeleton for study and extension**, not a claim of bit-accurate, production-ready synthesizable neural-network RTL. The project is intended for a Computer Science thesis portfolio where the emphasis is Python, PyTorch, graph transformation, quantization, software architecture and automated code generation.
