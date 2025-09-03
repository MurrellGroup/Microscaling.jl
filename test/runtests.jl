using Microscaling
using Test

using Einops
using Statistics

const FORMATS = [
    NVFP4,
    MXFP4,
    MXFP6_E2M3,
    MXFP6_E3M2,
    MXFP8_E4M3,
    MXFP8_E5M2,
    MXINT8,
]

@testset "Microscaling.jl" begin

    @testset "quantize" begin
        x = rand(Float32, 256)
        @testset for (formats, method) in [
            (FORMATS, GenericMethod()),
            ([MXFP4], MXFPMethod()),
        ]
            for format in formats
                z, y = quantize(x, format)
                k = size(y, 1) ÷ size(z, 1)
                @test size(y) == size(x)
                xb, yb = rearrange.((x, y), einops"(k n) -> k n"; k)
                zb = z'
                @test mean(abs, xb .- yb .* zb) < 1.0f0
            end
        end
    end

    @testset "bitpack" begin

        @testset "FP4" begin
            x = randn(Float32, 256)
            @testset for format in [NVFP4, MXFP4]
                z, y = quantize(x, format)
                y′ = bitpack(y)
                @test length(y′) == length(x) ÷ 2
                @test sizeof(y′) == sizeof(x) ÷ 8
            end
        end

    end

end
