using Printf
using Statistics

age = 12

if age >= 5 && age <= 6
    println("You're in Kindergarten")
elseif age >= 7 && age <= 13
    println("You're in Middle school")
elseif age >= 14 && age <= 18
    println("You're in High school")
else
    println("Stay Home")
end

@printf("true || false = %s\n", true || false ? "true" : "false")
@printf("!true = %s\n", !true ? "true" : "false")

# (조건) ? (참일 때) : (거짓일 때)