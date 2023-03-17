#=
mutable struct Gaussian
    Δt::Float64
    cwv::Float64 # Center wavelength
    spread::Float64 # spread
    peak_pos::Float64 # Peak location in time-domain.
end
=#

module Sources

using Printf
using Plots, Unitful
import PhysicalConstants.CODATA2018: c_0, μ_0, ε_0

c0 = ustrip(c_0)
μ0 = ustrip(μ_0)
ε0 = ustrip(ε_0)
THz = 1e12
nm = 1e-9

function gaussian(tstep, dt, cwv, spread, peak_pos)::Float64

    cfreq = c0 / cwv
    w0 = 2*π*cfreq
    ws = spread * w0
    ts = 1/ws
    tc = peak_pos * dt
    tt = tstep*dt - tc
    pulse_re = (exp((-.5) * ((tt/ts)^2)) * cos(w0*tt))

    return pulse_re
end

function gaussian_plotting(tsteps, dt, cwv, spread, peak_pos, wvlens)
    
    T = collect(0:1:tsteps)
    pulse = vec(gaussian.(T, dt, cwv, spread, peak_pos))
    # println(length(pulse))
    # println(size(pulse))
    freqs = reverse(c0 ./ wvlens)

    #println(size(vec(freqs)))
    #println(size(tsteps))

    # freqs and tsteps are vector. 
    # They should be converted to Matrix
    # before we operate outer product on them.
    #println(typeof(vec(freqs)))
    #println(typeof(tsteps))

    # freqs = reshape(freqs, length(freqs), 1)
    # tsteps = reshape(tsteps, 1, length(tsteps))

    # By reshape function, we can make an array to Matrix.
    #println(typeof(vec(freqs)))
    #println(typeof(tsteps))

    # println(size(vec(freqs) * tsteps))
    # println(size(exp.((1im*2*π*dt) .* (vec(freqs) * tsteps))))
    # println(size(freqs))
    # println(size(dt .* pulse'))
    # println(size(exp.((1im*2*π*dt) .* freqs .* T')))
    # println(size(dt .* pulse .* exp.((1im*2*π*dt) .* freqs' .* T)))

    pulse_ft = vec(sum(exp.((1im*2*π*dt).*freqs*T') * (dt.*pulse), dims=2))
    #=
    println(typeof(pulse_ft))
    println(size(pulse_ft))
    println(dimension(pulse_ft))
    =#

    pulse_ft_amp = abs.(pulse_ft)
    # println(typeof(pulse_ft_amp))
    # println(size(pulse_ft_amp))
    # println(dimension(pulse_ft_amp))
    # println(dimension(wvlens))

    plot_t = plot(T, pulse)
    # println(dimension(pulse))
    # plot_w = plot(wvlens, pulse_ft_amp, ylim=(0,maximum(pulse_ft_amp)))
    plot_w = plot(wvlens./nm, pulse_ft_amp)
    plot_f = plot(freqs./THz, pulse_ft_amp)

    fig = plot(plot_t, plot_w, plot_f, layout=(1,3), size=(1000,400))

    savefig(fig, "./gaussian.png")

end

if abspath(PROGRAM_FILE) == @__FILE__

    c0 = 299792458
    mu0 = 4*π*10^(-7)

    um = 1e-6
    nm = 1e-6

    Lx, Ly, Lz = 128*10*um, 128*10*um, 128*10*um
    Nx, Ny, Nz = 128, 128, 128
    dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz

    courant = 1/4
    dt = courant * min(dx,dy,dz) / c0
    Tsteps = 3000

    cwv = 300*um
    interval = 2
    spread   = 0.3
    peak_pos = 1000
    plot_per = 100

    wvlens = collect(200*um:interval*um:600*um)
    #println(wvlens)
    freqs = c0 ./ wvlens
    t = collect(0:Tsteps)

    #println(t)
    #println(t.*2)
    #println(typeof(t))
    gauss = gaussian(peak_pos, dt, cwv, spread, peak_pos)
    #println(gauss)
    gauss_plot = gaussian_plotting(t, freqs, dt, cwv, spread, peak_pos)
end

end