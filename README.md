# PyTorch Computational Graph Architecture Explorer

A software-only Python project for analysing PyTorch models with `torch.fx`.

The project focuses on computational-graph inspection, model-structure analysis, reproducible benchmarking, and comparison of software/model configurations. It does **not** generate RTL, SystemVerilog, FPGA/ASIC designs, or hardware implementation artifacts.

## Features

- Build or load small PyTorch models
- Trace models with `torch.fx`
- Extract operations and graph dependencies
- Capture tensor shapes where available
- Count trainable parameters
- Estimate multiply-accumulate operations for supported layers
- Compare model configurations using software-oriented metrics
- Export analysis results to JSON
- Run reproducible experiments with fixed random seeds
- Validate the analysis workflow with Pytest

## Supported layers

The current analyser recognises common PyTorch modules including:

- `nn.Linear`
- `nn.Conv2d`
- `nn.ReLU`
- `nn.Flatten`
- `nn.MaxPool2d`
- `nn.AdaptiveAvgPool2d`

Other FX graph nodes are preserved as generic operations where possible.

## Quick start

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
python train_demo.py
python main.py --model demo_mlp --input-shape 1,16
```

CNN example:

```bash
python main.py --model tiny_cnn --input-shape 1,1,28,28 --out generated
```

The analysis is written to:

```text
generated/model_analysis.json
```

## Example metrics

The report includes:

- graph node count
- operation types
- tensor shapes
- trainable parameter count
- estimated MACs for supported layers
- dependency relationships
- model output shape

## Validation

The current implementation has been validated against native PyTorch model statistics for both the built-in `DemoMLP` and `TinyCNN` examples.

Observed validation results:

- Automated test suite: **5/5 tests passed**
- `DemoMLP` parameter count: analyser **676**, PyTorch **676**
- `DemoMLP` estimated MACs: **640**
- `DemoMLP` graph nodes: **5**
- `DemoMLP` output shape: **[1, 4]**
- `TinyCNN` parameter count: analyser **1,316**, PyTorch **1,316**
- `TinyCNN` estimated MACs: **282,304**
- `TinyCNN` graph nodes: **10**
- `TinyCNN` output shape: **[1, 4]**
- Repeated analysis produces consistent structural results
- JSON report generation and dependency extraction are covered by automated tests

The validation is architecture-level rather than dataset-accuracy validation because this project analyses model structure rather than predictive performance.

## Project purpose

This repository is intended as a Computer Science portfolio project demonstrating:

- Python software development
- PyTorch and PyTorch FX
- computational-graph analysis
- structured intermediate representations
- model inspection
- reproducible experimentation
- automated testing
- JSON-based reporting

## Scope

This is a software-analysis project. It intentionally excludes RTL generation, SystemVerilog, fixed-point hardware design, FPGA/ASIC implementation, synthesis, place-and-route, and physical hardware deployment.
