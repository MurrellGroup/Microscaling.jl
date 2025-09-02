"""
    BlockFormat{E,S,k}

A block format specifies the element type, scale type, and number of elements per block.
"""
struct BlockFormat{E,S,k} end

floatrange(T::Type{<:AbstractFloat}) = -floatmax(T), floatmax(T)
convert_clamp(T, x) = convert(T, clamp(x, floatrange(T)...))

abstract type Method end
struct GenericMethod <: Method end
struct MXFPMethod <: Method end

# TODO: struct UnbiasedMXFP4Method <: Method end
# See algorithm 2 in https://arxiv.org/pdf/2502.20586

function quantize_blocks(
    ::GenericMethod, V::AbstractArray{T}, ::BlockFormat{E,S,k}
) where {T,E,S,k}
    size(V, 1) == k || error("Block format specifies $k elements per block, but input has $(size(V, 1)) elements")
    amax = maximum(abs, V, dims=1)
    X = S.(amax ./ E.(floatmax(E)))
    P = convert_clamp.(E, V ./ X)
    return X, P
end

function quantize_blocks(
    ::MXFPMethod, V::AbstractArray{T}, ::BlockFormat{E,S,k}
) where {T,E,S<:MX_E8M0,k}
    size(V, 1) == k || error("Block format specifies $k elements per block, but input has $(size(V, 1)) elements")
    emax_elem = Microfloats.right_aligned_exponent(floatmax(E)) - Microfloats.bias(E)
    shared_exp = floor.(Int16, log2.(maximum(abs, V, dims=1))) .- Int16(emax_elem)
    X = MX_E8M0(2.0f0 .^ shared_exp)
    P = convert_clamp.(E, V ./ X)
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
    method=GenericMethod(), axis=:column
) where {E,S,k}
    V = if axis === :column
        rearrange(V, einops"(k n) ... -> k n ..."; k)
    elseif axis === :row
        rearrange(V, einops"m (k n) ... -> k n m ..."; k)
    else
        error("Invalid axis: $axis, must be :column or :row")
    end
    X, P = quantize_blocks(method, V, format)
    return rearrange.((X, P), einops"k n ... -> (k n) ..."; k)
end
