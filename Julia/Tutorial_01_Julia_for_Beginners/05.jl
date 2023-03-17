x = 1
y = 2

if x>y
    println("$x is greater than $y")
elseif x <y
    println("$x is less than $y")
else
    println("$x is equal to $y")
end

#= A Ternaray operator is an operator that takes three arguments.
Julia uses ? and :.

a ? b : c

if a is true, b is executed.
if a is false, c is executed.
Note that spaces between ? and :. It is very important.
=#

x > y ? println("$x > $y") : println("$x ≤ $y")

i = 1

while i <= 10
    println(i)
    global i += 1
end

i = 1

while true
    println(i)
    if i>=5
        break
    end
    global i += 1
    
end
#=
begin
    print("You're in the Lost Forest. Go left of right?")
    i = lowercase(readline())
    while i == "right"
        print("You're in the Lost Forest. Go left of right?")
        global i = lowercase(readline())
    end
    println("You got out of the Lost Forest.")
end
=#

for i in 1:10
    println(i)
end

 for i in 5:15
    println(i)
 end

 for i in 10:-1:1
    println(i)
 end

 for char in 'a':2:'z'
    println(char)
 end

 for greek in 'α':'ω'
    println(greek)
 end

 for i in 1:127
    println(i, "\t", Char(i))
 end

 d1 = Dict("A" => 1, "B"=> π, "C" => "doggo")
 for (key, value) in d1
    println("key = $key\tvalue = $value")
 end

 d2 = Dict()
 
 for i in 1:10
    d2[i] = i^2
 end

 tp1 = (1, π, "doggo")

 for i in tp1
    println(i)
 end

 a1 = [1, π, "doggo"]

 for i in a1
    println(i)
 end

 for i in 1:10
    push!(a1, i)
 end

 # Let's write FizzBuzz in Julia!
 for i in 1:100
    if i % 3 == 0 && i % 5 ==0
        println("FizzBuzz")
    elseif i %3 == 0
        println("Fizz")

    elseif i %5 == 0
        println("Buzz")
    else
        println(i)
    end
end

x, y = 4, 5

A = fill(0, (x,y))

for i in 1:x
    for j in 1:y
        A[i, j] = i+ j
    end
end

println(A)

B = fill(0, (x,y))

# Syntactic Sugar example.
for i in 1:x, j in 1:y
    B[i,j] = i+j
end

println(B)

C = [i+j for i in 1:x, j in 1:y]
println(C)

for d1 in 1:6
    for d2 in 1:6
        counter = d2 + 6 * (d1 - 1)
        println("$counter\t$d1 + $d2 = $(d1+d2)")
    end
end

for d1 in 1:6
    for d2 in 1:6
        if (d1+d2)%3 == 0
            d2 <= 3 ? counter = d1 *2 -1 : counter = d1 * 2 
            println("$counter\t$d1 + $d2 = $(d1+d2)")
        end
    end
end

s = "Hello, world!\n"

for i in s
    println(i)
end

begin
    an_letters = "aefhilmnorsxAEFHILMNORSX"
    print("\nI will cheer for you! Enter a word: ")
    word = "hey"
    print("Enthusiasm level. Enter a number between 1 and 10: ")
    times = 3 
    println()
    for i in word
        shout = uppercase(i)
        if i in an_letters
            print("Give me an\t$i !"); sleep(1)
            println("\t$shout !!!"); sleep(0.5)
        else
            print("Give me a\t$i !"); sleep(1)
            println("\t$shout !!!"); sleep(0.5)
        end
    end
    println("\nWhat does that spell?\n"); sleep(1)
    for j in 1:times
        println(word, " !!!"); sleep(0.25)
    end
end

# Examples of Comprehension.
data = [i for i in 1:5]
data = [i^2 for i in 1:5]
data = [i*j for i in 1:5, j in 5:10]
data = [i for i in 1:10 if 1%2 == 1]
data = [[i for i in 1:10] [j for j in 11:20] [k for k in 21:30]]
d3 = Dict("$i"=>i for i in 1:10 if i % 2 == 1)

mycolors = [:black, :blue, :cyan, :green, :yellow, :magenta, :red, :white]

for color in mycolors
    printstyled(s, bold=true, color=color)
end