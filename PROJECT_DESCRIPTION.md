# Project: PyTorch Computational Graph Architecture Explorer

## Problem

Machine-learning models can become difficult to inspect as their computational structure grows. Developers often need a clear view of operations, dependencies, tensor shapes, parameter counts, and computational cost before comparing or optimizing model configurations.

## Objective

Build a Python software tool that traces PyTorch models with `torch.fx`, converts the traced graph into a structured intermediate representation, and generates reproducible analysis reports.

## Main components

- PyTorch model definitions
- `torch.fx` graph tracing
- operation and dependency extraction
- tensor-shape inspection
- parameter counting
- MAC estimation for supported layers
- structured JSON reporting
- reproducible experiment configuration
- automated Pytest validation

## Evaluation

The project can compare several PyTorch architectures using:

- graph node count
- operation coverage
- parameter count
- estimated MACs
- output tensor shape
- dependency structure
- repeatability of generated reports

## Scope

This repository is software-only. It does not perform RTL generation, SystemVerilog generation, hardware mapping, fixed-point hardware design, FPGA/ASIC synthesis, or physical deployment.
