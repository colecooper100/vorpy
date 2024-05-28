using Plots
using Statistics
include(string(pwd(), "/julia_src/src/utility_functions/utility_functions.jl"))



###### Biot-Savart solver function ######
print("Importing Biot-Savart solver... ")
t0 = time_ns()  # TIMING
include(string(pwd(), "/julia_src/weighted_biot_savart_solver_cpu.jl"))

# Set the solver function
function bsfn(fps, vpps, cdms, circs; kwargs...)
    # Change this to change the solver
    return weighted_biot_savart_solver_cpu(fps, vpps, cdms, circs; kwargs...)
end

println("Done ", elapsed_time(t0), " seconds.")  # TIMING
println()  # Blank line



###### Specify vortex and test parameters ######
NUMSEGS = 2000
RINGCENTER = (0, 0, 0)
RINGRADIUS = 1
CORERADIUS = 0.01
CIRCULATION = 1.0
println("Making vortex ring...")
println("* Number of segments: ", NUMSEGS)
println("* Ring center: ", RINGCENTER)
println("* Ring radius: ", RINGRADIUS)
println("* Core radius: ", CORERADIUS)
println("* Circulation: ", CIRCULATION)
println()  # Blank line



###### Generate the vortex ring and plot ######
# Generate the vortex points
theta = range(0, 2 * pi, length=NUMSEGS+1)
vpx = RINGRADIUS .* cos.(theta) .+ RINGCENTER[1]
vpy = RINGRADIUS .* sin.(theta) .+ RINGCENTER[2]
vpz = zeros(size(vpx)...) .+ RINGCENTER[3]
vpps = stack([vpx, vpy, vpz], dims=1)
# To prevent numerical errors, explicitly set
# the last point to the the first.
vpps[:, end] = vpps[:, 1]

# Generate the core radii
crads = ones(NUMSEGS+1) * CORERADIUS
crads[end] = crads[1]

# Generate the circulation
circs = ones(NUMSEGS+1) * 1.0
circs[end] = circs[1]

# Plot the ring
vorplt = plot(vpps[1, :], vpps[2, :], vpps[3, :])
title!("Vortex ring")
xlabel!("x")
ylabel!("y")
zlabel!("z")
display(vorplt)



###### Analytical solution for vortex ring ######
#=============================================
This result comes from "Vortex rings in
classical and quantum systems" by C F Barenghi
and R J Donnelly from Fluid Dynamics Research.
This is only for the velocity on the vortex
ring and assumes the ring radius >> core
radius. Additionally, the vorticity
distribution in the core is assumed is assumed
to be Gaussian.
=============================================#
# This result applies only to points on the
# vortex ring and the velocity is uniform, so
# the function does not take field points or
# the position of the ring.
function analytical_solution_vortex_ring(circulation, ringradius, coreradius)
    term1 = circulation / (4 * pi * ringradius)
    term2 = log(8 * ringradius / coreradius)
    return term1 * (term2 - 0.558)
end

# Print analytical solution
print("Ring velocity (analytical): ")
println(analytical_solution_vortex_ring(CIRCULATION,
                                            RINGRADIUS,
                                            CORERADIUS))

println()  # Blank line



###### Numerically compute the vortex ring velocity ######
println("Computing numerical solution for vortex ring...")
t0 = time_ns()  # TIMING
num_vel = bsfn(vpps, vpps, crads, circs; verbose=false)
println("* Numerical solution done ", elapsed_time(t0), " seconds.")  # TIMING
println("* Maximum velocity (z-axis): ", maximum(num_vel[3, :]))
println("* Minimum velocity (z-axis): ", minimum(num_vel[3, :]))
println("* Average velocity (z-axis): ", mean(num_vel[3, :]))
println("* Ring velocity (numerical):")
display(num_vel)

# Compute the error
rmserr, errvec = RMSerror(num_vel, ana_vel_fin_lo)
println("* Total RMS Error: ", rmserr)
# println("Size of anavel_lo: ", size(anavel_lo))  # DEBUG
# println("Size of num_vel: ", size(num_vel))  # DEBUG
# println("Size of errvec: ", size(errvec))  # DEBUG

println()  # Blank line