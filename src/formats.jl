"""
    BlockFormat{E,S,k}

A block format specifies the element type, scale type, and number of elements per block.
"""
abstract type BlockFormat{E,S,k} end

"""
    NVFP4

`NVFP4` is a microscaling format using FP4 elements (E2M1, no NaN/Inf),
with E4M3 scale factors, each scaling contiguous element blocks of 16.
"""
struct NVFP4 <: BlockFormat{MX_E2M1, Float8_E4M3, 16} end

"""
    MXFP4

`MXFP4` is a microscaling format using FP4 elements (E2M1, no NaN/Inf),
with E8M0 scale factors, each scaling contiguous element blocks of 32.
"""
struct MXFP4 <: BlockFormat{MX_E2M1, MX_E8M0, 32} end

"""
    MXFP6_E2M3

`MXFP6_E2M3` is a microscaling format using FP6 elements (E2M3, no NaN/Inf),
with E8M0 scale factors, each scaling contiguous element blocks of 32.
"""
struct MXFP6_E2M3 <: BlockFormat{MX_E2M3, MX_E8M0, 32} end

"""
    MXFP6_E3M2

`MXFP6_E3M2` is a microscaling format using FP6 elements (E3M2, no NaN/Inf),
with E8M0 scale factors, each scaling contiguous element blocks of 32.
"""
struct MXFP6_E3M2 <: BlockFormat{MX_E3M2, MX_E8M0, 32} end

"""
    MXFP8_E4M3

`MXFP8_E4M3` is a microscaling format using FP8 elements (E4M3, including NaN),
with E8M0 scale factors, each scaling contiguous element blocks of 32.
"""
struct MXFP8_E4M3 <: BlockFormat{MX_E4M3, MX_E8M0, 32} end

"""
    MXFP8_E5M2

`MXFP8_E5M2` is a microscaling format using FP8 elements (E5M2, including NaN/Inf),
with E8M0 scale factors, each scaling contiguous element blocks of 32.
"""
struct MXFP8_E5M2 <: BlockFormat{MX_E5M2, MX_E8M0, 32} end

"""
    MXINT8

`MXINT8` is a microscaling format using INT8 elements (Int8, scaled down by 64),
with E8M0 scale factors, each scaling contiguous element blocks of 32.
"""
struct MXINT8 <: BlockFormat{Q1f6, MX_E8M0, 32} end
