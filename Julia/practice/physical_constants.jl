import PhysicalConstants.CODATA2018 as phy
#import PhysicalConstants.CODATA2018: c_0, μ_0, ε_0
using Printf
#using Formatting

#eps = float(Float64, phy.VacuumElectricPermittivity)
#mu = float(Float64, phy.VacuumMagneticPermeability)
#eps = PhysicalConstants.CODATA2018
#mu = PhysicalConstants.MKS.VacuumMagneticPermeability
eps = phy.VacuumElectricPermittivity
mu = phy.VacuumMagneticPermeability

println(mu)
print("\n\n")
println(eps)

println(float(Float64, mu))
println(typeof(float(Float64, mu)))
#print(μ_0)
#@printf(".3f", c_0)
#println("$eps", typeof(eps))
#println("$mu", typeof(eps))
#printfmt("{:.5f}", mu)
#@printf("%f", mu)
#fmt.printfmtln("{:5.3f}", eps)
#fmt.printfmtln("{:f}", mu)