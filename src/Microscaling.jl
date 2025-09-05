module Microscaling

using Einops
using FixedPointNumbers
using Microfloats

include("formats.jl")
export BlockFormat
export NVFP4
export MXFP4
export MXFP6_E2M3, MXFP6_E3M2
export MXFP8_E4M3, MXFP8_E5M2
export MXINT8

include("layouts.jl")
export MatrixLayout, RowMajor, ColumnMajor
export ScaleFactorLayout, Naive, Sm1xx

include("quantize.jl")
export quantize
export dequantize

include("bitpack/bitpack.jl")
export bitpack, bitpack!
export bitunpack, bitunpack!

include("blockscaled.jl")
export BlockScaled
export NaiveBlockScaled
export Sm1xxBlockScaled

end
