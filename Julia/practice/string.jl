using Printf
using Statistics

s1 = "Just some random words\n"
println(length(s1))

println(s1[1])
println(s1[end])
println(s1[1:4])


s2 = string("Yukiteru", "Amano")

println(s2)
println("Yuno"*" Gasai")

i3 = 2
i4 = 3
println("$i3+$i4 = $(i3+i4)")

s3 = """
I
have
many
lines"""

println("Takao">"Hiyama")
println(findfirst(isequal('i'), "Keigo"))
println(occursin("key", "monkey"))