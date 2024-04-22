using Plots
include("../src/timing.jl")

###### Import BS solver ######
print("Importing Biot-Savart solver... ")

t0 = time_ns()  # TIMING
include("../weighted_biot_savart_kernel_cpu.jl")
println("Done ", elapsed_time(t0), " seconds.")  # TIMING

println()  # Blank line


###### Set Biot-Savart solver function ######
bsfn(fps, vpps, cdms, circs; kwargs...) = weighted_biot_savart_solver_cpu(fps, vpps, cdms, circs; kwargs...)


###### Specify vortex and test parameters ######
ENDPOINTS_LO = [-1000, 1000]
NUMSEGS_LO = 2  # was 10
COREDIAMETER_LO = 0.0001  # was 5.0
CIRCULATION_LO = 10.0
ZPLANE_LO = 0.0
NUMVELSAMP_LO = 50
println("Making Lamb-Oseen Vortex...")
println("* Endpoints: ", ENDPOINTS_LO)
println("* Number of segments: ", NUMSEGS_LO)
println("* Core diameter: ", COREDIAMETER_LO)
println("* Circulation: ", CIRCULATION_LO)
println("* Z-plane: ", ZPLANE_LO)
println("* Number of velocity samples: ", NUMVELSAMP_LO)
println()  # Blank line


###### Utility functions ######
function ERR(arr1, arr2)
    # println("Size of arr1: ", size(arr1))  # DEBUG
    # println("Size of arr2: ", size(arr2))  # DEBUG
    err = arr1 .- arr2
    # println("Size of err_vec: ", size(err))  # DEBUG
    RMSerr = sqrt.(sum(err .^ 2) / length(err))
    return err, RMSerr
end

# Define analytical solution function
# I would like this to be a continuous function
# so any position in space can be given and the
# velocity at that point can be computed. But,
# for now, we will just compute the velocity at
# a set point along the z-axis.
# I think to make this a continuous function,
# you would step through all the points on the
# vortex path and compute the distance of the
# path point to the field point, the two path
# points which are the closest to the field
# point would be,,,
# Didn't I already solve this with the Biot-Savart
# law? 
function analytical_solution_lo(fps, corediameters, circulations)
    x = fps[1, :]
    y = fps[2, :]
    z = fps[3, :]
    # println("Size of fps: ", size(fps))  # DEBUG
    # println("Size of x: ", size(x))  # DEBUG
    # println("Size of y: ", size(y))  # DEBUG
    # println("Size of z: ", size(z))  # DEBUG

    cylindrical_radius = sqrt.(x .^ 2 .+ y .^ 2)
    azimuthal_angle = atan.(y, x)
    term1 = circulations[1] ./ (2 .* pi .* cylindrical_radius)
    term2 = exp.(-(cylindrical_radius .^ 2) ./ (2 * corediameters[1]^2))
    # println("Length of azimuthal_angle: ", length(azimuthal_angle))  # DEBUG
    # println("azimuthal_angle: ", azimuthal_angle)  # DEBUG
    # println("Size of term1: ", size(term1))  # DEBUG
    # println("Size of term2: ", size(term2))  # DEBUG

    direction = Array{Float64, 2}(undef, 3, length(azimuthal_angle))
    direction[1, :] .= -sin.(azimuthal_angle)
    direction[2, :] .= cos.(azimuthal_angle)
    direction[3, :] .= zeros(length(azimuthal_angle))
    term1 = reshape(term1, 1, length(term1))
    term2 = reshape(term2, 1, length(term2))

    # println("Size of direction: ", size(direction))  # DEBUG
    # finite solution, infinite solution
    return term1 .* (1 .- term2) .* direction, term1 .* direction
end


###### Make Lamb-Oseen vortex ######
# Make the vortex path
# Align the vortex with the z-axis
vpps_lo = zeros(3, NUMSEGS_LO + 1)
vpps_lo[3, :] .= range(ENDPOINTS_LO[1], stop=ENDPOINTS_LO[2], length=NUMSEGS_LO + 1)

# Define the core diameters
# Eventually I would like to allow the core diameter
# to vary along the path, but for now, we will just
# use a constant diameter.
crdms_lo = ones(NUMSEGS_LO + 1) .* COREDIAMETER_LO

# Define the circulations
# Eventually I would like to allow the circulations
# to vary along the path, but for now, we will just
# use a constant circulation.
circs_lo = ones(NUMSEGS_LO + 1) .* CIRCULATION_LO


###### Compute and plot the analytical solution ######
println("Making 2D velocity plot of analytical solution...")

# Make some field points to plot the analytical
# solution
fps_lo = zeros(3, 8)
fps_lo[1, :] .= [1, 0.707, 0, -0.707, -1, -0.707, 0, 0.707]
fps_lo[2, :] .= [0, 0.707, 1, 0.707, 0, -0.707, -1, -0.707]

# Compute the analytical solution
velocities_lo = Array{Float64, 2}(undef, size(fps_lo))
for i in axes(fps_lo, 2)
    vel_lo = analytical_solution_lo(reshape(fps_lo[:, i], 3, 1), crdms_lo, circs_lo)
    velocities_lo[:, i] .= vel_lo[1]  # store finite solution
end

# Make a quiver plot of the analytical solution
quiver_lo = quiver(fps_lo[1, :], fps_lo[2, :], quiver=(velocities_lo[1, :], velocities_lo[2, :]), legend=false)
title!("Velocity of Lamb-Oseen Vortex\n(Analytical Solution)")
xlabel!("x")
ylabel!("y")
display(quiver_lo)

println()  # Blank line


###### Compute and compare the velocity profile ######
println("Comparing numerical solution to analytical...")

# Make the field points
xvec_lo = range(0.1, 40, length=NUMVELSAMP_LO)
yvec_lo = range(0.1, 40, length=NUMVELSAMP_LO)
zvec_lo = ones(length(xvec_lo)) .* ZPLANE_LO
fps_lo = stack([xvec_lo, yvec_lo, zvec_lo], dims=1)
# println("Size of fps_lo: ", size(fps_lo))  # DEBUG

# Compute analytical solution
print("* Computing analytical solution... ")
t0 = time_ns()  # TIMING
ana_vel_fin_lo, ana_vel_inf_lo = analytical_solution_lo(fps_lo, crdms_lo, circs_lo)
println("Done ", elapsed_time(t0), " seconds.")  # TIMING

# Compute numerical solution
println("* Computing numerical solution... ")
t0 = time_ns()  # TIMING
bsvel_lo = bsfn(fps_lo, vpps_lo, crdms_lo, circs_lo; verbose=true)
println("* Numerical solution done ", elapsed_time(t0), " seconds.")  # TIMING

# Compute the error
err_lo, RMSerr_lo = ERR(bsvel_lo, ana_vel_fin_lo)
println("* Total RMS Error: ", RMSerr_lo)
# println("Size of anavel_lo: ", size(anavel_lo))  # DEBUG
# println("Size of bsvel_lo: ", size(bsvel_lo))  # DEBUG
# println("Size of err_lo: ", size(err_lo))  # DEBUG

println()  # Blank line

# Plot the error
println("Generating plot of the error...")
errplt_lo = plot(xvec_lo, abs.(err_lo[1, :]), label="x-error")
plot!(xvec_lo, abs.(err_lo[2, :]), label="y-error")
plot!(xvec_lo, abs.(err_lo[3, :]), label="z-error")
title!("Error of Velocity Profile\nat z=$(ZPLANE_LO)")
xlabel!("x")
ylabel!("Absolute Error")
display(errplt_lo)

# Plot the velocity profile
println()  # Blank line
println("Generating plot of the velocity profile...")
velprofplt_lo = plot(xvec_lo, ana_vel_fin_lo[2, :], label="Analy sol finite")
plot!(xvec_lo, ana_vel_inf_lo[2, :], label="Analy sol infinite")
vline!([COREDIAMETER_LO, COREDIAMETER_LO*5, COREDIAMETER_LO*10], label="Core Diameters (1, 5x, 10x)")
plot!(xvec_lo, bsvel_lo[2, :], label="Biot-Savart solution")
title!("Velocity Profile of Lamb-Oseen Vortex\nat z=$(ZPLANE_LO)")
xlabel!("x")
ylabel!("y")
# ylims!(0, 0.6)
display(velprofplt_lo)