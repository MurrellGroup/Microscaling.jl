using BFloat16s

struct MicroscaledArray{F<:BlockFormat,Axis,T,N,Element,Scale} <: AbstractArray{T,N}
    element::Element
    scale::Scale
end

const ColumnMajorMicroscaledArray{F} = MicroscaledArray{F,1}
const RowMajorMicroscaledArray{F} = MicroscaledArray{F,2}

block_size(::MicroscaledArray{F}) where F = block_size(F)
contiguous_axis(::MicroscaledArray{<:Any,Axis}) where Axis = Axis

function MicroscaledArray{F,Axis,T}(
    element::Element, scale::Scale
) where {F,Axis,T,Element<:AbstractArray,Scale<:AbstractArray}
    return MicroscaledArray{F,Axis,T,ndims(element)-1,Element,Scale}(element, scale)
end

function MicroscaledArray{F,Axis}(
    element::Element, scale::Scale
) where {F,Axis,Element<:AbstractArray,Scale<:AbstractArray}
    T = promote_type(eltype(element), eltype(scale))
    return MicroscaledArray{F,Axis,T}(element, scale)
end

Base.size(μ::MicroscaledArray) = move((prod(size(μ.element)[1:2]), size(μ.element)[3:end]...), 1, contiguous_axis(μ))

function Base.getindex(μ::MicroscaledArray, I::Int...)
    J = move(I, contiguous_axis(μ), 1)
    i, j... = J
    index_in_block = mod1(i, block_size(μ))
    block_index = cld(i, block_size(μ))
    return μ.element[index_in_block, block_index, j...] * μ.scale[1, block_index, j...]
end

Base.print_array(io::IO, μ::MicroscaledArray) = Base.print_array(io, Base.broadcastable(μ))

function BitPacking.bitpacked(x::MicroscaledArray{F,Axis}) where {F,Axis}
    return MicroscaledArray{F,Axis}(bitpacked(x.element), x.scale)
end

function BitPacking.bitunpacked(x::MicroscaledArray{F,Axis}) where {F,Axis}
    return MicroscaledArray{F,Axis}(bitunpacked(x.element), x.scale)
end

Base.copy(μ::MicroscaledArray) = MicroscaledArray(copy(μ.element), copy(μ.scale))

Base.similar(μ::MicroscaledArray, ::Type{T}, dims::Dims=size(μ)) where T = similar(μ.scale, BFloat16, dims)

### Broadcasting

Broadcast.broadcastable(μ::MicroscaledArray) = dequantize(μ)

struct MicroscaledArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end

function Base.copyto!(μ::MicroscaledArray{F,Axis,T}, bc::Broadcast.Broadcasted) where {F,Axis,T}
    quantize!(μ, Broadcast.materialize(bc))
    return μ
end

function Base.materialize!(μ::MicroscaledArray{F,Axis,T}, bc::Broadcast.Broadcasted) where {F,Axis,T}
    quantize!(μ, Broadcast.materialize(bc))
    return μ
end

using FillArrays

# doesn't work for 0
function Base.materialize!(μ::MicroscaledArray{F,Axis,T}, bc::Broadcast.Broadcasted{<:Broadcast.AbstractArrayStyle{0}}) where {F,Axis,T}
    v, = bc.args
    quantize!(μ, Fill(v, block_size(μ)))
    return μ
end
