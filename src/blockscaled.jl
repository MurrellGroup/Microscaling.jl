struct BlockScaled{F<:BlockFormat,M<:MatrixLayout,S<:ScaleFactorLayout,T1,T2}
    format::F
    matrix_layout::M
    scale_factor_layout::S
    scale::T1
    element::T2
    num::Int
    dim::Int
end

function BlockScaled(format, matrix_layout, scale_factor_layout, X, P)
    block_size, block_count, num = size(X)
    dim = block_size * block_count
    return BlockScaled(format, matrix_layout, scale_factor_layout, X, P, num, dim)
end

const NaiveBlockScaled{F,M} = BlockScaled{F,M,Naive}
const Sm1xxBlockScaled{F,M} = BlockScaled{F,M,Sm1xx}
