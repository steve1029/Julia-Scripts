using Printf
using Statistics

mutable struct TypeExample
    a::Int64
    b::Float64
    #c = zeros(Float64, 5)
end

x = TypeExample(3,4)
println(x.a)
println(x.b)

function multiplier(xx::TypeExample)
   xx.a *= 2 
   xx.b *= 2
   println(xx.a)
end

multiplier(x)


c2 = Char(120)
println(c2)

i1 = UInt8(trunc(3.14))
println(i1)

f1 = parse(Float64, "1")
println(i1)

i2 = parse(Int8, "1")
println(i2)