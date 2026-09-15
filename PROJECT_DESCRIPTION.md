# Project: PyTorch Model-to-RTL Architecture Explorer

## Problem
AI deployment on embedded accelerators requires understanding model structure, numeric precision, memory cost and computational parallelism before hardware implementation.

## Adapted objective
Build a Python software tool that converts PyTorch models into a compiler-style intermediate representation and automatically produces fixed-point simulation results, architecture estimates, an RTL hierarchy, and SystemVerilog scaffolding.

## Main software components
- PyTorch model loader
- `torch.fx` graph parser
- Layer/shape/parameter/MAC analyzer
- Fixed-point quantization simulator
- Parallelism and latency estimator
- SystemVerilog template generator
- JSON hierarchy/build manifests
- Markdown planning report
- Automated tests

## Difference from the Ericsson thesis
This implementation intentionally stops before FPGA synthesis, place-and-route, bitstream generation and physical deployment. It is therefore suitable as a software/Computer Science portfolio prototype rather than a reproduction of Ericsson's internal thesis work.

## Possible thesis evaluation
Compare several PyTorch architectures using:
- graph conversion coverage
- parameter and MAC extraction accuracy
- quantization error
- estimated memory usage
- idealized cycle count under different parallelism
- generated-code coverage by layer type
