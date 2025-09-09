using TupleTools: insertafter, permute

val(x) = x isa Val ? x : Val(x)
unval(::Val{x}) where x = x
unval(x) = x

function moveperm(n, src, dest)
    return TupleTools.insertafter(
        (ntuple(identity, src-1)..., ntuple(i -> i+src, n-src)...),
        dest-1, (src,)
    )::NTuple{n}
end

moveperm(::Val{n}, args...) where n = moveperm(n, args...)
moveperm(::NTuple{n}, args...) where n = moveperm(n, args...)

moveinvperm(x, src, dest) = moveperm(x, dest, src)

function moveaxis(x::A, ::Val{src}, ::Val{dest}) where {T,N,A<:AbstractArray{T,N},src,dest}
    src === dest && return x
    perm = moveperm(N, src, dest)
    iperm = moveinvperm(N, dest, src)
    return PermutedDimsArray{T,N,perm,iperm,A}(x)
end

function moveaxis(x::AbstractArray, src, dest)
    return permutedims(x, moveperm(N, src, dest))
end

move(x::Tuple, args...) = permute(x, moveperm(x, args...))
