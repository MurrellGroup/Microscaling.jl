struct MicroscaledArray{F<:BlockFormat,Axis,T,N,Element,Scale} <: AbstractArray{T,N}
    element::Element
    scale::Scale
end

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

function microscaled(x::AbstractArray, format::F; T=eltype(x), axis=Val(1)) where F<:BlockFormat
    permuted_size = move(size(x), unval(axis), 1)
    element_size = (block_size(format), permuted_size[1] ÷ block_size(format), permuted_size[2:end]...)
    element = similar(x, T, element_size...)
    scale = similar(x, T, 1, element_size[2:end]...)
    μ = MicroscaledArray{F,unval(axis),T}(element, scale)
    quantize!(μ, x)
    return μ
end

Base.size(μ::MicroscaledArray) = permute((size(μ.element, 1) * size(μ.element, 2), size(μ.element)[3:end]...), axis_invperm(μ))

function Base.getindex(μ::MicroscaledArray, I::Int...)
    J = move(I, contiguous_axis(μ), 1)
    i, j... = J
    index_in_block = mod1(i, block_size(μ))
    block_index = cld(i, block_size(μ))
    return μ.element[index_in_block, block_index, j...] * μ.scale[1, block_index, j...]
end

Base.print_array(io::IO, μ::MicroscaledArray) = Base.print_array(io, Base.broadcastable(μ))

const ColumnMajorMicroscaledArray{F} = MicroscaledArray{F,1}
const RowMajorMicroscaledArray{F} = MicroscaledArray{F,2}

function BitPacking.bitpacked(x::MicroscaledArray{F,Axis}) where {F,Axis}
    return MicroscaledArray{F,Axis}(bitpacked(x.element), x.scale)
end

function BitPacking.bitunpacked(x::MicroscaledArray{F,Axis}) where {F,Axis}
    return MicroscaledArray{F,Axis}(bitunpacked(x.element), x.scale)
end

Base.copy(μ::MicroscaledArray) = MicroscaledArray(copy(μ.element), copy(μ.scale))


### Broadcasting

function Broadcast.broadcastable(μ::MicroscaledArray)
    x = reshape(μ.element .* μ.scale, :, size(μ.element)[3:end]...)
    return moveaxis(x, Val(1), Val(contiguous_axis(μ)))
end

struct MicroscaledArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end

Broadcast.BroadcastStyle(::Type{<:MicroscaledArray{F,Axis,T,N}}) where {F,Axis,T,N} = MicroscaledArrayStyle{N}()
(::Type{<:MicroscaledArrayStyle})(::Val{N}) where N = MicroscaledArrayStyle{N}()

function Base.copyto!(μ::MicroscaledArray{F,Axis,T}, bc::Broadcast.Broadcasted) where {F,Axis,T}
    quantize!(μ, Broadcast.materialize(bc))
    return μ
end

using FillArrays

# doesn't work for 0
function Base.copyto!(μ::MicroscaledArray{F,Axis,T}, bc::Broadcast.Broadcasted{<:Base.Broadcast.AbstractArrayStyle{0}}) where {F,Axis,T}
    v, = bc.args
    quantize!(μ, Fill(v, block_size(μ)))
    return μ
end
