bitpack(w) = (args...; kws...) -> bitpack(w, args...; kws...)
bitunpack(w) = (args...; kws...) -> bitunpack(w, args...; kws...)

bitpack(w::Int, args...; kws...) = bitpack(Val(w), args...; kws...)
bitunpack(w::Int, args...; kws...) = bitunpack(Val(w), args...; kws...)

include("chunk.jl")

bitpack(w::Val, x::NTuple{M,NTuple{N,UInt8}}) where {M,N} = map(bitpack(w), x)
bitunpack(w::Val, x::NTuple{M,NTuple{N,UInt8}}) where {M,N} = map(bitunpack(w), x)

# optimized for CUDA arrays
autogroup(::Val) = Val(1)
autogroup(::Val{1}) = Val(2)
autogroup(::Val{2}) = Val(8)
autogroup(::Val{3}) = Val(2)
autogroup(::Val{4}) = Val(8)
autogroup(::Val{5}) = Val(1)
autogroup(::Val{6}) = Val(2)
autogroup(::Val{7}) = Val(1)

half_val(::Val{M}) where M = Val(cld(M, 2))

chunk_size(::Val{W}) where W = abs(lcm(8, W) ÷ W)
packed_chunk_size(::Val{W}) where W = abs(lcm(8, W) ÷ 8)

function bitpack!(w::Val{W}, packed_bytes::AbstractArray{UInt8}, bytes::AbstractArray{UInt8}; groups::Val{M}=autogroup(w)) where {W,M}
    N = chunk_size(w)
    B = packed_chunk_size(w)
    M <= 16 || error("The number of groups must be less than or equal to 16")
    size(bytes, 1) % (M * N) == 0 ||
        error("The number of bytes in the first dimension ($(size(bytes, 1))) must be divisible by $N times the number of groups ($M). " *
              "The maximum allowed number of groups is 16, but fewer groups are often more efficient, especially on GPUs. " *
              "To change the number of groups, call `bitpack(...; groups=Val(M))`")
    chunks = reinterpret(NTuple{M,NTuple{N,UInt8}}, bytes)
    packed_chunks = reinterpret(NTuple{M,NTuple{B,UInt8}}, packed_bytes)
    map!(bitpack(w), packed_chunks, chunks)
    return packed_bytes
end

function bitpack(w::Val{W}, bytes::AbstractArray{UInt8}; kws...) where W
    packed_bytes = similar(bytes, size(bytes, 1) * W ÷ 8, size(bytes)[2:end]...)
    bitpack!(w, packed_bytes, bytes; kws...)
    return packed_bytes
end

function bitunpack!(w::Val{W}, bytes::AbstractArray{UInt8}, packed_bytes::AbstractArray{UInt8}; groups::Val{M}=half_val(autogroup(w))) where {W,M}
    N = chunk_size(w)
    B = packed_chunk_size(w)
    M <= 16 || error("The number of groups must be less than or equal to 16")
    size(packed_bytes, 1) % (M * N) == 0 ||
        error("The number of packed bytes in the first dimension ($(size(packed_bytes, 1))) must be divisible by $N times the number of groups ($M). " *
              "The maximum allowed number of groups is 16, but fewer groups are often more efficient, especially on GPUs. " *
              "To change the number of groups, call `bitunpack(...; groups=Val(M))`")
    packed_chunks = reinterpret(NTuple{M,NTuple{B,UInt8}}, packed_bytes)
    chunks = reinterpret(NTuple{M,NTuple{N,UInt8}}, bytes)
    map!(bitunpack(w), chunks, packed_chunks)
    return bytes
end

function bitunpack(w::Val{W}, packed_bytes::AbstractArray{UInt8}; kws...) where W
    bytes = similar(packed_bytes, size(packed_bytes, 1) * 8 ÷ W, size(packed_bytes)[2:end]...)
    bitunpack!(w, bytes, packed_bytes; kws...)
    return bytes
end

bitpack(::Val{8}, bytes::AbstractArray{UInt8}; kws...) = bytes
bitunpack(::Val{8}, bytes::AbstractArray{UInt8}; kws...) = bytes

function bitpack(w::Val, x::AbstractArray{T}; kws...) where T
    sizeof(T) === 1 || error("Bitpacking only supported for 8-bit types")
    bytes = reinterpret(UInt8, x)
    packed_bytes = bitpack(w, bytes; kws...)
    return packed_bytes
end

function bitunpack(w::Val, packed_bytes::AbstractArray{UInt8}, ::Type{T}; kws...) where T
    sizeof(T) === 1 || error("Bitpacking only supported for 8-bit types")
    bytes = bitunpack(w, packed_bytes; kws...)
    x = reinterpret(T, bytes)
    return x
end
