# import PhysicalConstants.CODATA2018 as phy
import PhysicalConstants.CODATA2018: c_0, ε_0, μ_0
using Unitful
# import Printf
include("./Sources.jl")
# import .Sources
# c = phy.SpeedOfLightInVacuum
# eps = phy.VacuumElectricPermittivity
# mu = phy.VacuumMagneticPermeability

nm = 1e-9
um = 1e-6
lunit = nm

c0 = ustrip(c_0)
μ0 = ustrip(μ_0)
ε0 = ustrip(ε_0)

# Space and time control.
tsteps = 1000
Nz = 100
dz = 10 * lunit
courant = 1/2
dt = courant * dz / c0

# Source control.
cwv = 200*lunit
w0 = (2*π*c0) / cwv
spread =  0.08
peak_pos = 500
interval = 1*lunit
w1 = w0 * (1-spread*2)
w2 = w0 * (1+spread*2)
l1 = 2*π*c0 / w1
l2 = 2*π*c0 / w2
wvlens = collect(l2:interval:l1)

# Field control.
Ex = zeros(Float64, Nz)
Hy = similar(Ex, Nz) # 'similar' function copies the type of the array.
Lz = (Nz-1) * dz

Sources.gaussian_plotting(tsteps, dt, cwv, spread, peak_pos, wvlens)

for t=0:1:tsteps
    pulse_re = Sources.gaussian(t, dt, cwv, spread, peak_pos)
    # println(pulse_re)
end