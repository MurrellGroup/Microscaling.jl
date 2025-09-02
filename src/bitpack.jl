bitwidth(T::Type) = sizeof(T) * 8
bitwidth(T::Type{<:Microfloat}) = Microfloats.n_utilized_bits(T)

function bitpack(A::Vector{V}, T::Type{<:Unsigned}=UInt8) where V
    elements_per_byte = bitwidth(T) ÷ bitwidth(V)
    storage = Vector{T}(undef, length(A) ÷ elements_per_byte)
    for i in 1:length(storage)
        x = zero(T)
        for j in 1:elements_per_byte
            x |= reinterpret(T, A[(i - 1) * elements_per_byte + j]) << ((j - 1) * bitwidth(V))
        end
        storage[i] = x
    end
    return storage
end
