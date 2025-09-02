# Microscaling

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/Microscaling.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/Microscaling.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/Microscaling.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/Microscaling.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/Microscaling.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/Microscaling.jl)

Microscaling is a Julia package that implements microscaling formats, quantization functions, and bitpacking functions,
based on the [Open Compute Project Microscaling Formats (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)

## Exported Formats

- `NVFP4`
- `MXFP4`
- `MXFP6_E2M3`
- `MXFP6_E3M2`
- `MXFP8_E4M3`
- `MXFP8_E5M2`
- `MXINT8`

## Installation

```julia
using Pkg
Pkg.Registry.add(url="https://github.com/MurrellGroup/MurrellGroupRegistry")
Pkg.add("Microscaling")
```
