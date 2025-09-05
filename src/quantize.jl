"""
    BlockFormat{E,S,k}

A block format specifies the element type, scale type, and number of elements per block.
"""
abstract type BlockFormat{E,S,k} end

floatrange(T) = -floatmax(T), floatmax(T)
convert_clamp(T, x) = convert(T, clamp(x, floatrange(T)...))

abstract type Method end
abstract type GenericMethod <: Method end
abstract type MXFPMethod <: Method end
# TODO: algorithm 2 in https://arxiv.org/pdf/2502.20586

function quantize_blocks(
    ::Type{GenericMethod}, V::AbstractArray{T}, ::Type{BlockFormat{E,S,k}}
) where {T,E,S,k}
    size(V, 1) == k || error("Block format specifies $k elements per block, but input has $(size(V, 1)) elements")
    X = @. S(maximum(abs, V, dims=1) / E(floatmax(E)))
    P = @. convert_clamp(E, V / X)
    return X, P
end

function quantize_blocks(
    ::Type{MXFPMethod}, V::AbstractArray{T}, ::Type{BlockFormat{E,S,k}}
) where {T,E,S<:MX_E8M0,k}
    size(V, 1) == k || error("Block format specifies $k elements per block, but input has $(size(V, 1)) elements")
    emax_elem = Microfloats.right_aligned_exponent(floatmax(E)) - Microfloats.bias(E)
    X = @. MX_E8M0(2.0f0 ^ floor(Int16, log2($maximum(abs, V, dims=1))) - Int16(emax_elem))
    P = @. convert_clamp(E, V / X)
    return X, P
end

"""
    quantize(V::AbstractArray, format::BlockFormat{E,S,k}; method=GenericMethod(), axis=:column)

Quantize the input array `V` to the given block format `format`.

# Arguments
- `V::AbstractArray`: The input array to quantize.
- `format::BlockFormat{E,S,k}`: The block format to quantize to.
- `method::Method`: The method to use for quantization.
- `axis::Symbol`: The axis to quantize along. Must be `:column` or `:row`.
If `:row`, the first two dimensions are transposed such that the blocks are contiguous along the first dimension.

# Returns
- `X::AbstractArray`: The scale factors.
- `P::AbstractArray`: The quantized values.
"""
function quantize(
    V::AbstractArray, format::BlockFormat{E,S,k};
    method=GenericMethod, axis=:column
) where {E,S,k}
    V_blocks = if axis === :column
        rearrange(V, einops"(k n) ... -> k n ..."; k)
    elseif axis === :row
        rearrange(V, einops"m (k n) ... -> k n m ..."; k)
    else
        error("Invalid axis: $axis, must be :column or :row")
    end
    return quantize_blocks(method, V_blocks, format)
end

function dequantize(
    T::Type, X::AbstractArray{S}, P::AbstractArray{E}, ::Type{BlockFormat{E,S,k}};
    axis=:column
) where {E,S,k}
    V_blocks = T.(P)
    V_blocks .*= X
    V = if axis === :column
        rearrange(V_blocks, einops"k n ... -> (k n) ..."; k)
    elseif axis === :row
        rearrange(V_blocks, einops"k n m ... -> m (k n) ..."; k)
    else
        error("Invalid axis: $axis, must be :column or :row")
    end
    return V
end
