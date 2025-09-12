module Microscaling

using BitPacking
using Einops
using FixedPointNumbers
using Microfloats

include("utils.jl")

include("formats.jl")
export BlockFormat, block_size
export NVFP4
export MXFP4
export MXFP6_E2M3, MXFP6_E3M2
export MXFP8_E4M3, MXFP8_E5M2
export MXINT8

include("layouts.jl")
export ScaleFactorLayout, Naive, Sm1xx
export change_layout

include("MicroscaledArray.jl")
export MicroscaledArray
export ColumnMajorMicroscaledArray, RowMajorMicroscaledArray
export bitpacked

include("quantization.jl")
export quantize, quantize!
export dequantize, dequantize!

include("gemm.jl")

end
