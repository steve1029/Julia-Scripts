# install a 'Plots' package.
# DifferentialEquations.jl is a package for solving numerically solving differential equations in Julia by Chris Rackauckas.
# For more info, see https://docs.juliaplots.org/latest/ecosystem/#ecosystem
# and https://github.com/SciML/DifferentialEquations.jl
#=
import Pkg
Pkg.add("Plots")
Pkg.rm("DifferentialEquations")
=#
using Plots
x = 1:10;
y = rand(10,3);
println(size(y))

#fig1 = plot(x,y)
#savefig(fig1,"./test.png")
#display(fig1)
#sleep(5)
#readline()

x = 1:10
y = rand(1:10, (10,2))
println(y)

fig2 = plot(x, y)
savefig(fig2,"./test2.png")

y1 = rand(10)
y2 = rand(10)
fig3 = plot(x, [y1, y2, (1 ./x).^2])
savefig(fig3,"./test3.png")