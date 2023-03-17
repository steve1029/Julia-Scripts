x = 1//2
println(typeof(x))
println(typeof(π))
println(10^6 == 1000000 == 1_000_000)
println(sqrt(4))
println(cbrt(8))
println(convert(Int64, 3.0))

round(3.55, digits=1, RoundUp)
round(3.55, digits=1, RoundDown)

a = 3
b = 3.0

println(a == b)
println(a === b) # check if the type of the second operand is equal to that of the first operand.

typeof('a') # character.
typeof('#') # Google the 'julia unicode'. You can see many unicode defined in julia.

println('ℯ')
println(√3)
println(∛8)
println(0.1+0.2 ≈ 0.3) # \approx unicode. That's is fascinating!
E="mc²" # \backslash \caret 2 yields c^2 !!
H₂O = "water"
α = 1
println(Char(65)) # It gives you the ASCII code of the number.

s2 ="""This is an "interesting" tutorial."""
print(s2)

place = "Krusty Krab"
println("I am eating lunch at the $place.")

kpatty = 1.50
cbits = 1.25
sfsoda = 1.25
kmeal = 3.50

println("Bought separately = $(kpatty+cbits+sfsoda) dollars.")
println("\$100")
println("€100")

print("Enter a number: "); text = readline() # readline is the same as Input() in python.

snum = string(123)
println(typeof(snum))
println(typeof(true))
println(typeof(false))
println(typeof(Any))

color = :mycolor
println(typeof(color)) # julia has special type that is called 'Symbol' which is not defined in python.

s1 = "Hello, world!\n"
printstyled(s1, bold=true, color= :red)
printstyled(s1, bold=true, color= :magenta)
printstyled(s1, bold=true, color= :green)