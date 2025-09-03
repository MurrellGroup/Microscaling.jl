module Microscaling

using Einops
using FixedPointNumbers
using Microfloats

include("quantize.jl")
export quantize
export BlockFormat
export GenericMethod
export MXFPMethod

include("bitpack.jl")
export bitpack

const NVFP4 = BlockFormat{MX_E2M1, Float8_E4M3, 16}()
const MXFP4 = BlockFormat{MX_E2M1, MX_E8M0, 32}()
const MXFP6_E2M3 = BlockFormat{MX_E2M3, MX_E8M0, 32}()
const MXFP6_E3M2 = BlockFormat{MX_E3M2, MX_E8M0, 32}()
const MXFP8_E4M3 = BlockFormat{MX_E4M3, MX_E8M0, 32}()
const MXFP8_E5M2 = BlockFormat{MX_E5M2, MX_E8M0, 32}()
const MXINT8 = BlockFormat{Q1f6, MX_E8M0, 32}()

export NVFP4
export MXFP4
export MXFP6_E2M3, MXFP6_E3M2
export MXFP8_E4M3, MXFP8_E5M2
export MXINT8

"""
    NVFP4

`NVFP4` is a microscaling format using FP4 elements (E2M1, no NaN/Inf),
with E4M3 scale factors, each scaling contiguous element blocks of 16.
"""
NVFP4

"""
    MXFP4

`MXFP4` is a microscaling format using FP4 elements (E2M1, no NaN/Inf),
with E8M0 scale factors, each scaling contiguous element blocks of 32.
"""
MXFP4

"""
    MXFP6_E2M3

`MXFP6_E2M3` is a microscaling format using FP6 elements (E2M3, no NaN/Inf),
with E8M0 scale factors, each scaling contiguous element blocks of 32.
"""
MXFP6_E2M3

"""
    MXFP6_E3M2

`MXFP6_E3M2` is a microscaling format using FP6 elements (E3M2, no NaN/Inf),
with E8M0 scale factors, each scaling contiguous element blocks of 32.
"""
MXFP6_E3M2

"""
    MXFP8_E4M3

`MXFP8_E4M3` is a microscaling format using FP8 elements (E4M3, including NaN),
with E8M0 scale factors, each scaling contiguous element blocks of 32.
"""
MXFP8_E4M3

"""
    MXFP8_E5M2

`MXFP8_E5M2` is a microscaling format using FP8 elements (E5M2, including NaN/Inf),
with E8M0 scale factors, each scaling contiguous element blocks of 32.
"""
MXFP8_E5M2

"""
    MXINT8

`MXINT8` is a microscaling format using INT8 elements (Int8, scaled down by 64),
with E8M0 scale factors, each scaling contiguous element blocks of 32.
"""
MXINT8

end
