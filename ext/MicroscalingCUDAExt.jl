module MicroscalingCUDAExt

using Microscaling
using CUDA
using CUDA: BFloat16, CuPtr

using BitPacking
using Microfloats

const lib = "/home/anton/dev/mixedprec-gemm/build/libmixedprec_gemm.so"

@enum mpgemm_status_t::Cint begin
    MPGEMM_STATUS_SUCCESS = 0
    MPGEMM_STATUS_INVALID_ARGUMENT = 1
    MPGEMM_STATUS_UNSUPPORTED_SIZE = 2
    MPGEMM_STATUS_CUDA_ERROR = 3
    MPGEMM_STATUS_CUTLASS_ERROR = 4
    MPGEMM_STATUS_WORKSPACE_TOO_SMALL = 5
end

@enum mpgemm_elt_t::Cint begin
    MP_ELT_MX_E2M1 = 0
    MP_ELT_MX_E2M3 = 1
    MP_ELT_MX_E3M2 = 2
    MP_ELT_MX_E4M3 = 3
    MP_ELT_MX_E5M2 = 4
    MP_ELT_NV_E2M1 = 5
    MP_ELT_BF16    = 6
end

@enum mpgemm_sf_t::Cint begin
    MP_SF_MX_E8M0  = 0
    MP_SF_NV_UE4M3 = 1
end

@enum mpgemm_layout_t::Cint begin
    MP_ROW = 0
    MP_COL = 1
end

@enum mpgemm_mode_t::Cint begin
    MP_MODE_GEMM = 0
    MP_MODE_BATCHED = 1
end

const ELT_MAP = Dict{DataType,mpgemm_elt_t}(
    MX_E2M1 => MP_ELT_MX_E2M1,
    MX_E2M3 => MP_ELT_MX_E2M3,
    MX_E3M2 => MP_ELT_MX_E3M2,
    MX_E4M3 => MP_ELT_MX_E4M3,
    MX_E5M2 => MP_ELT_MX_E5M2,
    BFloat16 => MP_ELT_BF16,
)

const SF_MAP = Dict{DataType,mpgemm_sf_t}(
    MX_E8M0 => MP_SF_MX_E8M0,
    Float8_E4M3 => MP_SF_NV_UE4M3,
)

# Alignment in elements; choose conservative (>= required) values
const ALIGN_MAP = Dict{DataType,Cint}(
    MX_E2M1 => 32,
    MX_E2M3 => 128,
    MX_E3M2 => 128,
    MX_E4M3 => 16,
    MX_E5M2 => 16,
    BFloat16 => 8,
)

# Derive descriptor enums from array types
elt(::AbstractArray{T}) where T = ELT_MAP[T]
elt(::MicroscaledArray{<:BlockFormat{E}}) where E = ELT_MAP[E]
# Disambiguate NV vs MX for FP4 by format tag
elt(::MicroscaledArray{<:NVFP4}) = MP_ELT_NV_E2M1

sf(::MicroscaledArray{<:BlockFormat{<:Any,S}}) where {S} = SF_MAP[S]

layout(::ColumnMajorMicroscaledArray) = MP_COL
layout(::RowMajorMicroscaledArray) = MP_ROW
layout(::CuArray) = MP_COL

align(::MicroscaledArray{<:BlockFormat{E}}) where E = ALIGN_MAP[E]
align(::AbstractArray{E}) where E = ALIGN_MAP[E]

sv(x::MicroscaledArray) = block_size(x)

struct mpgemm_desc_t
    m::Cint; n::Cint; k::Cint

    eltA::mpgemm_elt_t; eltB::mpgemm_elt_t; eltC::mpgemm_elt_t; eltD::mpgemm_elt_t
    sfA::mpgemm_sf_t;  sfB::mpgemm_sf_t
    svA::Cint;  svB::Cint

    layoutA::mpgemm_layout_t; layoutB::mpgemm_layout_t; layoutC::mpgemm_layout_t; layoutD::mpgemm_layout_t
    alignA::Cint;  alignB::Cint;  alignC::Cint;  alignD::Cint

    # fix scale factor layout

    enable_epilogue_sfd::Cint
end

ismatrix(x) = x isa AbstractMatrix

function Microscaling._batched_gemm!(
    D::AbstractArray{BFloat16},
    A::MicroscaledArray,
    B::MicroscaledArray;
    alpha=1.0f0, beta=0.0f0,
    stream=CUDA.stream()
)
    C = D

    m, n = size(D)
    k = size(A, 2)
    @assert k == size(B, 1)

    desc = mpgemm_desc_t(
        m, n, k,

        #=MP_MODE_GEMM,
        max(size(A, 3), size(B, 3)),

        # For broadcast, use stride 0 when operand has no batch dim (matrix)
        # Element and scale factor arrays have size `blocks x M/blocks x K x B`
        ismatrix(A) ? 0 : stride(A_bytes, 4),
        ismatrix(B) ? 0 : stride(B_bytes, 4),
        ismatrix(C) ? 0 : stride(C, 3),
        ismatrix(D) ? 0 : stride(D, 3),
        ismatrix(A) ? 0 : stride(SFA, 4),
        ismatrix(B) ? 0 : stride(SFB, 4),=#

        elt(A), elt(B), elt(C), elt(D),
        sf(A), sf(B),
        sv(A), sv(B),

        layout(A), layout(B), layout(C), layout(D),
        align(A), align(B), align(C), align(D),

        false,
    )

    can_run = ccall((:mpgemm_can_run, lib), Cint, (Ref{mpgemm_desc_t},), desc)
    can_run == 0 && error("Kernel cannot run")

    A, B = bitpacked.((A, B)) # ensure bitpacked

    A_bytes, SFA = A.element.packed_bytes, A.scale
    B_bytes, SFB = B.element.packed_bytes, B.scale

    # TODO: do this in the workspace
    SFA_Sm1xx = change_layout(Naive() => Sm1xx(), SFA)
    SFB_Sm1xx = change_layout(Naive() => Sm1xx(), SFB)

    workspace_size = ccall((:mpgemm_workspace_size, lib), Csize_t, (Ref{mpgemm_desc_t},), desc)
    workspace = workspace_size > 0 ? CUDA.zeros(UInt8, workspace_size) : nothing
    workspace_ptr = isnothing(workspace) ? CuPtr{Cvoid}(0) : pointer(workspace)

    status = ccall((:mpgemm_execute, lib), mpgemm_status_t,
        (
            Ref{mpgemm_desc_t},
            CuPtr{Cvoid}, CuPtr{Cvoid},
            CuPtr{Cvoid}, CuPtr{Cvoid},
            CuPtr{Cvoid}, CuPtr{Cvoid},
            Cfloat, Cfloat,
            CuPtr{Cvoid}, Ptr{Cvoid},
        ),
        desc,
        pointer(A_bytes), pointer(SFA_Sm1xx),
        pointer(B_bytes), pointer(SFB_Sm1xx),
        pointer(C), pointer(D),
        alpha, beta,
        workspace_ptr, stream.handle
    )

    if status != MPGEMM_STATUS_SUCCESS
        msg = unsafe_string(ccall((:mpgemm_get_error_string, lib), Cstring, (mpgemm_status_t,), status))
        error(msg)
    end

    return D
end

end
