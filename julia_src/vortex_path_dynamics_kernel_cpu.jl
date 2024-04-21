using DifferentialEquations  # DE module
include("weighted_biot_savart_kernel_cpu.jl")
include("src/vortex_dormand_prince_method.jl")
# include("src/dormand_prince_method.jl")


# The function that _adaptive_dormand_prince
# solves.
_bsfndp(t, fps, vcrds, vcirs) = bs_solve_cpu(fps, fps, vcrds, vcirs)

# The function that DifferentialEquations
# solves.
# DE module
function _bsfnDE(vor_vel, vor_pos, params, t)
    # params = [vcrds, vcirs]
    # prob: r' = \frac{\Gamma}{4 \pi} \int_C \frac{\hat t \times \xi}{\|\xi\|^3} d\ell
    vor_vel .= bs_solve_cpu(vor_pos, vor_pos, params[1], params[2])
    return nothing
end

# DE module
function vortex_dynamics_DifferentialEquations(vpps_init, vcrds, vcirs, timespan, loc_err_tol; report_interval=nothing)
    prob = ODEProblem(_bsfnDE, vpps_init, timespan, [vcrds, vcirs])
    sol = solve(prob, DP5(), reltol=loc_err_tol, saveat=report_interval)
    vpps_evol = stack(sol.u)
    # println("Size of vpps_evol: ", size(vpps_evol))  # DEBUG
    vel_vec = Array{Float32, 3}(undef, size(vpps_evol))
    for i in axes(vpps_init, 3)
        vel_vec[:, :, i] .= bs_solve_cpu(vpps_init[:, :, i], vpps_init[:, :, i], vcrds, vcirs)
    end
    return vpps_evol, vel_vec, sol.t
end


################### DEBUGGING ###################
# function testfn(args...; kwargs...)
#     println("args: ", length(args), " ", args)
#     println("args[1]: ", size(args[1]), " ", args[1])
#     println("typeof(args[1]): ", typeof(args[1]))
#     println("kwargs: ", length(kwargs), " ", kwargs)
# end

using Plots
using BenchmarkTools


#================================================
Generate a vortex ring (points, core radii, and
circulation).
================================================#
# Vortex paramters
NUMSEGS = 100
RINGCENTER = (0, 0, 0)
RINGRADIUS = 5
CORERADIUS = 0.001
TSPAN = (0f0, 1f2)
LOCAL_ERR_TOL = 1f-2
println("Number of segments: ", NUMSEGS)
println("Ring center: ", RINGCENTER)
println("Ring radius: ", RINGRADIUS)
println("Core radius: ", CORERADIUS)
println("Time span: ", TSPAN)
println("Local error tolerance: ", LOCAL_ERR_TOL)

# Generate the vortex points
theta = range(0, 2 * pi, length=NUMSEGS+1)
cylin_radius = RINGRADIUS .* (1 .+ 3f-1 .* cos.(3 .* theta))
vpx = cylin_radius .* cos.(theta) .+ RINGCENTER[1]
vpy = cylin_radius .* sin.(theta) .+ RINGCENTER[2]
vpz = zeros(size(vpx)...) .+ RINGCENTER[3]
vpps = stack([vpx, vpy, vpz], dims=1)
# To prevent numerical errors, explicitly set
# the last point to the the first.
vpps[:, end] = vpps[:, 1]
println("STATUS: Generated vortex points.")

# display(plot(vpps[1, :], vpps[2, :], vpps[3, :], label="Vortex points"))

# Generate the core radii
vcrds = ones(NUMSEGS+1) * CORERADIUS
vcrds[end] = vcrds[1]
println("STATUS: Generated core radii.")

# Generate the circulation
vcirs = ones(NUMSEGS+1) * 1.0
vcirs[end] = vcirs[1]
println("STATUS: Generated circulations.")


# vortex_dynamics_adaptive_dormand_prince(vpps_init, vcrds, vcirs, timespan, min_step_size, max_step_size, loc_err_tol; report_interval=Inf32)
soldp, veldp, timedp = vortex_dynamics_adaptive_dormand_prince(vpps, vcrds, vcirs, TSPAN, LOCAL_ERR_TOL, 1f3, 1f-2; report_interval=1f2)
println("Size of soldp: ", size(soldp))

# vortex_dynamics_DifferentialEquations(vpps_init, vcrds, vcirs, timespan, loc_err_tol; report_interval=nothing)
solDE, velDE, timeDE = vortex_dynamics_DifferentialEquations(vpps, vcrds, vcirs, TSPAN, LOCAL_ERR_TOL; report_interval=1f2)
println("Size of solDE: ", size(solDE))

# L2 error in position
errplt = plot(transpose(sqrt.(sum((solDE[1, :, :] - soldp[1, :, :]).^2, dims=1) ./ length(timedp))), markershape=:xcross, label="x-position error")
plot!(errplt, transpose(sqrt.(sum((solDE[2, :, :] - soldp[2, :, :]).^2, dims=1) ./ length(timedp))), markershape=:circle, label="y-position error")
plot!(errplt, transpose(sqrt.(sum((solDE[3, :, :] - soldp[3, :, :]).^2, dims=1) ./ length(timedp))), markershape=:utriangle, label="z-position error")
title!("L2 Error in Position")
xlabel!("Time")
ylabel!("L2 Error")
display(errplt)

evolplt = plot()
for i in axes(soldp, 3)
    plot!(evolplt, soldp[1, :, i], soldp[2, :, i], soldp[3, :, i], label=timedp[i])
    plot!(evolplt, solDE[1, :, i], solDE[2, :, i], solDE[3, :, i], marker=:x, label=timeDE[i])
    # plt = plot(soldp[1, :, i], soldp[2, :, i], soldp[3, :, i], label=string("DP: ", timedp[i]))
    # plot!(solDE[1, :, i], solDE[2, :, i], solDE[3, :, i], label=string("DE: ", timeDE[i]))
    # display(plt)
end
display(evolplt)

