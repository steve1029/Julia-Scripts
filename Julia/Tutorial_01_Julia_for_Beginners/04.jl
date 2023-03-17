d1 = Dict()

println(typeof(d1))
d1 = Dict("A"=>1, "B"=> 3, "C"=>5, "D"=>7)
println(d1)

# Julia uses column-major order.

m = [] # array.
s = [1] # 1D array.
cv = [1,2,3] # column vector. It's dimension is 1.
println(typeof(cv))
rv = [1 2 3] # row vector. It's dimension is 2. So, in fact, it is a Matrix.
println(typeof(rv))

m = [1 3 5; 2 4 6] # 2D array is a Matrix.
println(typeof(m)) # Row x column

a1 = rand(4,2,3) # 3D is just an array. 
println(typeof(a1))

a = [1, 3.14, "doggo", :red]
println(typeof(a)) # You can put any type of data in an array.

# Julia indexing start from 1, not 0.
println(m[1, 2]) # row 1, column 2.
println(m[:, 3]) # column 3.
println(m[2, :]) # Row 2
println(m[1:2, 2:3])
println(m[end, :]) # Last row.
println(m[:, end]) # Last column.
println(m[:]) # Show all elements in sequence, regardless of the shape.

b = 3
m[1, 2] = 9 # :You can replace the elemens.
println(cv[2] +1)
println(sum(cv))
println(sum(rv))
println(cv .+ b)
println(size(m))
println(length(m))
println(eltype(m))
println(minimum(m))
println(maximum(m))
println(extrema(m)) # Shows min and max of the array.
println(transpose(m))
println(m') # A short hand systax of transpose.
println(reshape(m ,3, 2)) # reshape keeps your array column-major order, so it is different from transpose. Check yourself!
println(sort(cv, rev=true)) # sort function only works on column vector. It is not permanent.
println(cv) # Original vector is not changed.
println(sort!(cv, rev=true)) # For permanent sorting, you should use sort! function.
println(push!(cv,5)) # Add a 5 to the end of the column vector. It a permanent.
println(pop!(cv)) # Delete an element at the end of the column vector. It a permanent.

# There are some built-in functions helpful to populate arrays.
println(fill(π, 5, 5))
println(zeros(Int64, 3, 5))
println(ones(Float64, 3, 5))
println(trues(2, 8))
println(falses(7, 8))

A = [1,2,"eggdog"]
B = [1,2,"bongocat"]
println(1 in A)
println(1 ∈ A) # \in <TAB>
println(1 ∉ A) # \notin <TAB>
println(union(A, B)) # This creates an array with unique elements of A and B.
println(intersect(A, B)) # This creates an array with common elements of A and B.

a = [1,2,3]
b = [4,5,6]
c = [a, b] # An array of an array. It is not a matrix!
println(c)
println(typeof(c))
d = [a; b] # vertical concatenation.
vertical = vcat(a, b) # vertical concatenation.
e = [a b] # horizontal concatenation.
horizontal = hcat(a, b) # horizontal concatenation.

b = a
a[1] = 10
println(a)
println(b)

c = copy(a) # shallow copy.

d = Dict(1 => "apple", 2=> "banana")
tp = ("apple", "banana")
ntp = (one="apple", two="banana")
a = ["apple", "banana"]

varinfo() # Use it in REPL. tuple and named tuple consumes the least memory, than array and lastly, a dictionary.

rand()
rand(5)
rand(3,4)
rand(3,4,5)
rand(Float64, 3,4)
rand(Int64, 3,4)
rand(Bool, 3,4)
rand(1:10, 3,4)
rand(-10:10, 3,4)
rand(-10:5:100, 3,4)

s1 = "Hello, world!"
s1[1]
s1[11:-1:6]
typeof(s1)
typeof(s1[1])
lowercase(s1)
uppercase(s1)
reverse(s1)
occursin("hello", s1)
hooman = s1
doggo = replace(hooman, "hello"=> "henlo")
cv1 = collect(s1)
cv2 = split(s1, "")
cv3 = split(s1, " ")
cv4 = ["a", "b", "c"]
s2 = join(cvc4)
typeof(cvc4)
typeof(s2)
s3 = "Hello"
s3[1] = 'J' # It calls error. But the two methods below are possible.
s4 = "J" * s3[2:end]
s5 = replace(s3, "H" => "J")
reverse!(s1) # It call error. But the below is possible.
s6 = reverse(s1); s1 = s6