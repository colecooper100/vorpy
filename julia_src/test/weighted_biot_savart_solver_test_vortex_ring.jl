###### Import modules and local scrips ######
using Plots
using Statistics

# environment_variables.jl sets the path varibles
# used to call other scripts
# pwd() returns the "present working directory". For
# this project, pwd() should return the path to the
# vorpy directory
include(string(pwd(), "/julia_src/environment_variables.jl"))
include(string(UTILITY_FUNCTIONS, "/utility_functions.jl"))

# Biot-Savart solver
print("Importing Biot-Savart solver... ")
t0 = time_ns()  # TIMING
# MAKE SURE TO CHANGE THE SOLVER FOR bsfn() BELOW!!!!!
# include(string(VORPY, "/julia_src/weighted_biot_savart_solver_cpu.jl"))
include(string(VORPY, "/julia_src/weighted_biot_savart_solver_cuda.jl"))
println("Done ", elapsed_time(t0), " seconds.")  # TIMING
println()  # Blank line


###### Set the solver function ######
function bsfn(fps, vpps, crads, circs)
    # Change this to change the solver
    # DON'T FORGET TO INPUT THE CORRECT SCRIPT!!!!
    # return weighted_biot_savart_solver_cpu(fps, vpps, crads, circs)
    return weighted_biot_savart_solver_cuda(fps, vpps, crads, circs)
end


###### Specify vortex and test parameters ######
NUMSEGS = 2000
RINGCENTER = (Float32(0), Float32(0), Float32(0))
RINGRADIUS = Float32(1)
CORERADIUS = Float32(0.1)
CIRCULATION = Float32(1.0)
println("Making vortex ring...")
println("* Number of segments: ", NUMSEGS)
println("* Ring center: ", RINGCENTER)
println("* Ring radius: ", RINGRADIUS)
println("* Core radius: ", CORERADIUS)
println("* Circulation: ", CIRCULATION)
println()  # Blank line


###### Generate the vortex ring and plot ######
# Generate the vortex points
theta = collect(Float32, range(0, 2 * pi, length=NUMSEGS+1))
vpx = RINGRADIUS .* cos.(theta) .+ RINGCENTER[1]
vpy = RINGRADIUS .* sin.(theta) .+ RINGCENTER[2]
vpz = zeros(Float32, size(vpx)...) .+ RINGCENTER[3]
vpps = stack([vpx, vpy, vpz], dims=1)
# To prevent numerical errors, explicitly set
# the last point to the the first.
vpps[:, end] = vpps[:, 1]

# Generate the core radii
crads = ones(Float32, NUMSEGS+1) * CORERADIUS
crads[end] = crads[1]

# Generate the circulation
circs = ones(Float32, NUMSEGS+1) * Float32(1.0)
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
    term2 = log(8 * ringradius / (sqrt(2) * coreradius))
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
num_vel = bsfn(vpps, vpps, crads, circs)
println("* Numerical solution done ", elapsed_time(t0), " seconds.")  # TIMING
println("* Maximum velocity (z-axis): ", maximum(num_vel[3, :]))
println("* Minimum velocity (z-axis): ", minimum(num_vel[3, :]))
println("* Average velocity (z-axis): ", mean(num_vel[3, :]))
println("* Ring velocity (numerical):")
display(num_vel)

