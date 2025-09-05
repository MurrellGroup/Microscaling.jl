convert_clamp(T, x) = convert(T, clamp(x, -floatmax(T), floatmax(T)))

function quantize_generic(
    V::AbstractArray{T}, ::BlockFormat{E,S,k}
) where {T,E,S,k}
    size(V, 1) == k || error("Block format specifies $k elements per block, but input has $(size(V, 1)) elements")
    X = @. S(maximum(abs, V, dims=1) / E(floatmax(E)))
    P = @. convert_clamp(E, V / X)
    return X, P
end

function quantize_mxfp(
    V::AbstractArray{T}, ::BlockFormat{E,S,k}
) where {T,E,S<:MX_E8M0,k}
    size(V, 1) == k || error("Block format specifies $k elements per block, but input has $(size(V, 1)) elements")
    emax_elem = Microfloats.right_aligned_exponent(floatmax(E)) - Microfloats.bias(E)
    X = @. MX_E8M0(2.0f0 ^ floor(Int16, log2($maximum(abs, V, dims=1))) - Int16(emax_elem))
    P = @. convert_clamp(E, V / X)
    return X, P
end

# TODO: algorithm 2 in https://arxiv.org/pdf/2502.20586

to_blocks(V::AbstractArray, ::ColumnMajor) = rearrange(V, einops"(k n) ... -> k n ..."; k)
to_blocks(V::AbstractArray, ::RowMajor) = rearrange(V, einops"m (k n) ... -> k n m ..."; k)

from_blocks(V_blocks::AbstractArray, ::ColumnMajor) = rearrange(V_blocks, einops"k n ... -> (k n) ..."; k)
from_blocks(V_blocks::AbstractArray, ::RowMajor) = rearrange(V_blocks, einops"k n m ... -> m (k n) ..."; k)

"""
    quantize(V::AbstractArray, format::BlockFormat{E,S,k})

Quantize the input array to the given block format, returning the scale factors and quantized values.
"""
function quantize(
    V::AbstractArray, format::BlockFormat{E,S,k};
    algorithm=quantize_generic,
    layout=ColumnMajor,
) where {E,S,k}
    V_blocks = to_blocks(V, layout)
    X, P = algorithm(V_blocks, format)
    return X, P
end

function dequantize(
    T::Type, X::AbstractArray{S}, P::AbstractArray{E}, ::Type{BlockFormat{E,S,k}};
    layout=ColumnMajor,
) where {E,S,k}
    V_blocks = T.(P)
    V_blocks .*= X
    V = from_blocks(V_blocks, layout)
    return V
end
