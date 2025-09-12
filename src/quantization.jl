convert_clamp(T, x) = convert(T, clamp(x, -floatmax(T), floatmax(T)))

# TODO: algorithm 2 in https://arxiv.org/pdf/2502.20586

# FIXME: this generic version doesn't really work
function quantize_blocks!(μ::MicroscaledArray{<:BlockFormat{E,S}}, V::AbstractArray) where {E,S}
    @. μ.scale = S($maximum(abs, V, dims=1) / E(floatmax(E)))
    @. μ.element = convert_clamp(E, V / μ.scale)
    return μ
end

function quantize_blocks!(μ::MicroscaledArray{<:MXFPFormat{E}}, V::AbstractArray) where E
    @. μ.scale = MX_E8M0(2.0f0 ^ (floor(Int16, log2($maximum(abs, V, dims=1))) - Int16(exponent(floatmax(E)))))
    @. μ.element = convert_clamp(E, V / μ.scale)
    return μ
end

# could reshape before and then lazily move two axes at once
function quantize!(μ::MicroscaledArray, V::AbstractArray)
    V_contig = moveaxis(V, contiguous_axis(μ), 1)
    V_blocks = rearrange(V_contig, einops"(k n) ... -> k n ..."; k=block_size(μ))
    quantize_blocks!(μ, V_blocks)
    return μ
end

function quantize(x::AbstractArray, format::F; T=eltype(x), axis=Val(1)) where {E,S,F<:BlockFormat{E,S}}
    permuted_size = move(size(x), unval(axis), 1)
    element_size = (block_size(format), permuted_size[1] ÷ block_size(format), permuted_size[2:end]...)
    element = similar(x, E, element_size...)
    scale = similar(x, S, 1, element_size[2:end]...)
    μ = MicroscaledArray{F,unval(axis),T}(element, scale)
    quantize!(μ, x)
    return μ
end

function dequantize(μ::MicroscaledArray, T::Type=eltype(μ))
    V_blocks = T.(μ.element)
    V_blocks .*= μ.scale
    V_contig = rearrange(V_blocks, einops"k n ... -> (k n) ..."; k=block_size(μ))
    V = moveaxis(V_contig, Val(1), Val(contiguous_axis(μ)))
    return V
end

function dequantize!(V::AbstractArray{T}, μ::MicroscaledArray) where T
    V .= dequantize(μ, T)
    return V
end
