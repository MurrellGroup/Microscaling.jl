using LinearAlgebra

function _batched_gemm! end

function batched_gemm! end

function LinearAlgebra.mul!(
    C::AbstractMatrix,
    A::MicroscaledArray,
    B::MicroscaledArray,
    alpha::Number=1.0f0, beta::Number=0.0f0;
    kws...
)
    return _batched_gemm!(C, A, B; alpha, beta, kws...)
end
