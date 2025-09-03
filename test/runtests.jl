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

        x = rand(Float32, 128, 64)
        @testset for (format, method) in [
            (NVFP4, GenericMethod()),
            (MXFP4, GenericMethod()),
            (MXFP4, MXFPMethod()),
            (MXFP6_E2M3, GenericMethod()),
            (MXFP6_E3M2, GenericMethod()),
            (MXFP8_E4M3, GenericMethod()),
            (MXFP8_E5M2, GenericMethod()),
            (MXINT8, GenericMethod()),
        ]
            @testset "column" begin
                z, y = quantize(x, format; method, axis=:column)
                k = size(y, 1) ÷ size(z, 1)
                @test size(y) == size(x)
                xb = rearrange(x, einops"(k n) ... -> k n ..."; k)
                yb = rearrange(y, einops"(k n) ... -> k n ..."; k)
                zb = rearrange(z, einops"... -> 1 ...")
                @test mean(abs, xb .- yb .* zb) < 1.0f0
            end
            @testset "row" begin
                z, y = quantize(x, format; method, axis=:row)
                k = size(y, 1) ÷ size(z, 1)
                @test size(y) == size(rearrange(x, einops"a b ... -> b a ..."))
                xb = rearrange(x, einops"m (k n) ... -> k n m ..."; k)
                yb = rearrange(y, einops"(k n) m ... -> k n m ..."; k)
                zb = rearrange(z, einops"... -> 1 ...")
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
