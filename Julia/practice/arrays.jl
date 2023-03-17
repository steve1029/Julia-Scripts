using Printf
using Statistics

a1 = zeros(Int32, 2, 2)
a2 = Array{Int32}(undef, 5)
a3 = Float64[]
a4 = [1,2,3]

println(a4[1])
println(a4[end])
println(5 in a4)
println(findfirst(isequal(2), a4))

f(a) = (a >= 2) ? true : false

println(findall(f, a4))
println(count(f, a4))

#=
isequal(x)
Create a function that compares its argument to x using isequal, i.e. a function equivalent to y -> isequal(y, x).

The returned function is of type Base.Fix2{typeof(isequal)}, which can be used to implement specialized methods.
=#

println(size(a4)) # axis size
println(length(a4)) # array length
println(sum(a4))
splice!(a4, 2:1, [8,9])
println(a4)
splice!(a4, 2:3)
println(a4)
println(maximum(a4))
println(minimum(a4))
println(a4 * 2)
println(a4)

#= splice!(a::Vector, indices, [replacement]) -> items
Remove items at specified indices, and return a collection containing the removed items. Subsequent items are shifted left to fill the resulting gaps. If specified, replacement values from an ordered collection will be spliced in place of the removed items; in this case, indices must be a UnitRange.

To insert replacement before an index n without removing any items, use splice!(collection, n:n-1, replacement).

Julia 1.5 =#

println()
a5 = [1, 3.14, "Hello"]

a6 = [sin, cos, tan]
for n in a6
    println(n(0))
end

a7 = [1 2 3; 4 5 6]
for n = 1:2, m = 1:3
    @printf("%d ", a7[n,m])
end
println()
println()
println(a7[:, 2])
println(a7[2, :])

a8 = collect(1:5)
a9 = collect(2:2:10)
for n in a9 println(n) end

a10 = [n^2 for n in 1:5]
println(a10)

push!(a10, 36)

a11 = [n * m for n in 1:3, m in 1:4]
println(a11)

a12 = rand(0:9, 5, 5)
for n = 1:5
    for m = 1:5
        print(a12[n,m])
    end
    println()
end

print("\n\n")
a = [1 2 3]
b = [1.1 2.2 3.3]
c = a.*b
println(c)
println(3. .* a .* .5)
d = a .+ 1im .* b
println(d)
println(sum(abs.(d)))
println(sqrt(sum(abs2.(d))))

e = [a b]
println(e)

println(typeof(a))
println(typeof(a) == Matrix{Int})
println(typeof(a) === Matrix{Int})

println(typeof(3.))
println(typeof(3.) == Float64)
println(typeof(3.) == AbstractFloat)

println(typeof(3))
println(typeof(3) == Int)
println(typeof(3) == Int64)
println(typeof(3) == AbstractFloat)

outer = vec(a) * b
println(typeof(a))
println(typeof(b))
println(outer)
println(size(outer))