abstract type MatrixLayout end
struct RowMajor <: MatrixLayout end
struct ColumnMajor <: MatrixLayout end

abstract type ScaleFactorLayout end
struct Naive <: ScaleFactorLayout end
struct Sm1xx <: ScaleFactorLayout end

change_layout(::Pair{T,T}, x) where T<:ScaleFactorLayout = x
change_layout(::Pair{Naive,Sm1xx}, x) = rearrange(x, einops"1 (SF4 SFb) (M32 M4 Mb) -> SF4 M4 M32 SFb Mb"; SF4=4, M32=32, M4=4)
change_layout(::Pair{Sm1xx,Naive}, x) = rearrange(x, einops"SF4 M4 M32 SFb Mb -> 1 (SF4 SFb) (M32 M4 Mb)"; SF4=4, M32=32, M4=4)
