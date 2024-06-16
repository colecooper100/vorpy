###### Set the path to the `julia_fns` directory ######
# julia_env.jl sets global varibles used by
# other scrips in the `julia_fns` directory.
# This is an absolute path so it will be
# different on different systems. Thiis could
# also be set using relative paths but I
# don't want to do that right now.
JULIA_FNS = "/home/crashoverride/Dropbox/code/vorpy/julia_fns"
include(string(JULIA_FNS, "/julia_env.jl"))


###### Activate the Julia project environment ######
# This is needed because we are putting the project
# toml and manifest in the julia_fns directory but
# the REPL starts with the vorpy directory as the
# working directory.
using Pkg
Pkg.activate(JULIA_FNS)


###### Import modules and local scrips ######
using Plots

include(string(UTILITY_FUNCTIONS, "/utility_functions.jl"))

# Biot-Savart solver
print("Importing Biot-Savart solver... ")
t0 = time_ns()  # TIMING
# MAKE SURE TO CHANGE THE SOLVER FOR bsfn() BELOW!!!!!
# include(string(JULIA_FNS, "/weighted_biot_savart_solver_cpu.jl"))
include(string(JULIA_FNS, "/weighted_biot_savart_solver_cuda.jl"))
println("Done ", elapsed_time(t0), " seconds.")  # TIMING
println()  # Blank line


###### Set the solver function ######
function bsfn(fps, vpps, crads, circs)
    # Change this to change the solver
    # DON'T FORGET TO INPUT THE CORRECT SCRIPT!!!!
    # return weighted_biot_savart_solver_cpu(fps, vpps, crads, circs)
    return weighted_biot_savart_solver_cuda(fps, vpps, crads, circs)
end


###### Test and vortex parameters ######
ENDPOINTS = [-100000, 100000]
NUMSEGS = 100  # was 2
CORERADIUS = Float32(5)  # was 0.0001
CIRCULATION = Float32(10.0)
ZPLANE = Float32(0.0)
CONSTANTRADIUS = CORERADIUS * Float32(6)  # CORERADIUS / 4
NUMVELSAMP = 100
VELSAMPRANGE = [0.1, 40] # was [40, 100]
println("Making Lamb-Oseen Vortex...")
println("* Endpoints: ", ENDPOINTS)
println("* Number of segments: ", NUMSEGS)
println("* Core radius: ", CORERADIUS)
println("* Circulation: ", CIRCULATION)
println("* Z-plane: ", ZPLANE)
println("* Constant radius: ", CONSTANTRADIUS)
println("* Number of velocity samples: ", NUMVELSAMP)
println()  # Blank line


###### Analytical solution for Lamb-Oseen vortex ######
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
function analytical_solution_lamb_oseen(fp, coreradius, circulations)
    # FOR NOW WE ASSUME THE VORTEX IS ALIGNED WITH THE Z-AXIS
    # IN THE FUTURE, THIS REQUIREMENT WILL BE RELXEAD AND THE
    # FUNCTION WILL BE MADE TO WORK FOR ANY STRAIGHT VORTEX.
    x = fp[1]
    y = fp[2]
    z = fp[3]

    cylindrical_radius = sqrt.(x .^ 2 .+ y .^ 2)
    azimuthal_angle = atan.(y, x)
    term1 = circulations / (2 * Float32(pi) * cylindrical_radius)
    term2 = exp(-(cylindrical_radius^2) / (2 * coreradius^2))

    # direction = [sin.(azimuthal_angle), -cos.(azimuthal_angle), 0f0]  # this is wrong
    direction = [-sin.(azimuthal_angle), cos.(azimuthal_angle), 0f0]

    # Return: fat core solution, thin core solution
    return (term1 * (1 - term2)) .* direction, term1 .* direction
end


###### Make the vortex ######
# Make the vortex path
# Align the vortex with the z-axis
vpps = zeros(Float32, 3, NUMSEGS + 1)
vpps[3, :] .= range(ENDPOINTS[1], stop=ENDPOINTS[2], length=NUMSEGS + 1)

# Define the core radii
# Eventually I would like to allow the core radius
# to vary along the path, but for now, we will just
# use a constant radius.
crads = ones(Float32, NUMSEGS + 1) .* CORERADIUS

# Define the circulations
# Eventually I would like to allow the circulations
# to vary along the path, but for now, we will just
# use a constant circulation.
circs = ones(Float32, NUMSEGS + 1) .* CIRCULATION

# println("Vortex Path: ", vpps)  # DEBUG
# println("Core Radii: ", cor_rads)  # DEBUG
# println("Circulations: ", circs)  # DEBUG


###### Evaluate solutions at a constant distance ######
# This evaluate the solution all around the vortex
# at a constant radius from the vortex path.
println("Evaluate numerical solution at a constant distance...")

# Make some field points to plot (a circle around
# the vortex path)
fps = zeros(Float32, 3, 8)
theta = Float32.(range(0, 2*pi, length=8))
fps[1, :] .= CONSTANTRADIUS .* -sin.(theta)
fps[2, :] .= CONSTANTRADIUS .* cos.(theta)

# Compute the analytical solution
ana_vels_fat_core = Array{Float32, 2}(undef, size(fps)...)
ana_vels_thin_core = Array{Float32, 2}(undef, size(fps)...)
print("* Computing analytical solution... ")
t0 = time_ns()  # TIMING
for i in axes(fps, 2)
    # (I believe) the analytical solution works only for
    # constant core size and and circulation. So, we give
    # the first core size and circulation.
    vel_fat_core, vel_thin_core = analytical_solution_lamb_oseen(fps[:, i], crads[1], circs[1])
    ana_vels_fat_core[:, i] .= vel_fat_core
    ana_vels_thin_core[:, i] .= vel_thin_core
end
println("Done ", elapsed_time(t0), " seconds.")  # TIMING

# Compute the numerical solution
println("* Computing numerical solution... ")
t0 = time_ns()  # TIMING
num_vels = bsfn(fps, vpps, crads, circs)
println("* Numerical solution done ", elapsed_time(t0), " seconds.")  # TIMING

# 2D quiver plot of the velocity
plt_vrtx_fps = scatter([vpps[1, 1]], [vpps[2, 1]], marker=:x, label="Vortex Path")
scatter!(fps[1, :], fps[2, :], marker=:circle, label="Field Points")
quiver!(fps[1, :], fps[2, :], quiver=(ana_vels_fat_core[1, :], ana_vels_fat_core[2, :]), label="Analytical Solution (Fat Core)")
quiver!(fps[1, :], fps[2, :], quiver=(num_vels[1, :], num_vels[2, :]), label="Numerical Solution")
title!("Velocity of Lamb-Oseen Vortex\n(Constant Distance)")
xlabel!("x")
ylabel!("y")
display(plt_vrtx_fps)

# 3D quiver plot of the velocity
plt_vrtx_fps = plot(vpps[1, :], vpps[2, :], vpps[3, :], marker=:x, label="Vortex Path")
scatter!(fps[1, :], fps[2, :], fps[3, :], marker=:circle, label="Field Points")
quiver!(fps[1, :], fps[2, :], fps[3, :], quiver=(ana_vels_fat_core[1, :], ana_vels_fat_core[2, :], ana_vels_fat_core[3, :]), label="Analytical Solution (Finite Core)")
quiver!(fps[1, :], fps[2, :], fps[3, :], quiver=(num_vels[1, :], num_vels[2, :], num_vels[3, :]), label="Numerical Solution")
title!("Velocity of Lamb-Oseen Vortex (Constant Distance)")
xlabel!("x")
ylabel!("y")
zlabel!("z")
display(plt_vrtx_fps)

# Compute and plot the error
num_vels_mag = sqrt.(sum(num_vels .^ 2, dims=1))
println("Max magnitude of velocity (numerical): ", max(num_vels_mag...))
println("Min magnitude of velocity (numerical): ", min(num_vels_mag...))
rmserr, errvec = RMSerror(num_vels, ana_vels_fat_core)
println("* Total RMS Error (constant distance): ", rmserr)
errplt = plot(axes(fps, 2), abs.(errvec[1, :]), label="x-error")
plot!(axes(fps, 2), abs.(errvec[2, :]), label="y-error")
plot!(axes(fps, 2), abs.(errvec[3, :]), label="z-error")
title!("Error of Velocity (Constant Distance)\nat z=$(ZPLANE)")
xlabel!("x")
ylabel!("Absolute Error")
display(errplt)

println()  # Blank line



###### Compute and compare the velocity profile ######
# The velocity is computed as a function of distance 
# from the vortex path
println("Comparing numerical solution to analytical...")

# Make the field points
fpxvec = collect(Float32, range(VELSAMPRANGE..., length=NUMVELSAMP))
fpyvec = collect(Float32, range(VELSAMPRANGE..., length=NUMVELSAMP))
fpzvec = ones(Float32, length(fpxvec)) .* ZPLANE
fps = stack([fpxvec, fpyvec, fpzvec], dims=1)
# println("Size of fps: ", size(fps))  # DEBUG

# Compute analytical solution
print("* Computing analytical solution... ")
t0 = time_ns()  # TIMING
ana_vels_fat_core = zeros(size(fps)...)
ana_vel_thin_core = zeros(size(fps)...)
for i in axes(fps, 2)
    vel_fat_core, vel_thin_core = analytical_solution_lamb_oseen(fps[:, i], crads[1], circs[1])
    ana_vels_fat_core[:, i] .= vel_fat_core
    ana_vel_thin_core[:, i] .= vel_thin_core
end
println("Done ", elapsed_time(t0), " seconds.")  # TIMING

# Compute numerical solution
println("* Computing numerical solution... ")
t0 = time_ns()  # TIMING
num_vels = bsfn(fps, vpps, crads, circs)
println("* Numerical solution done ", elapsed_time(t0), " seconds.")  # TIMING

# Compute and plot the error
rmserr, errvec = RMSerror(num_vels, ana_vels_fat_core)
println("* Total RMS Error: ", rmserr)
errplt = plot(fpxvec, abs.(errvec[1, :]), label="x-error")
plot!(fpxvec, abs.(errvec[2, :]), label="y-error")
plot!(fpxvec, abs.(errvec[3, :]), label="z-error")
vline!([CORERADIUS, CORERADIUS*5, CORERADIUS*10], label="Core radii (1, 5x, 10x)")
title!("Error of Velocity Profile\nat z=$(ZPLANE)")
xlabel!("x")
ylabel!("Absolute Error")
display(errplt)

# Plot the velocity profile
println()  # Blank line
println("Generating plots of the velocity profile...")
axeslabel = ["x", "y", "z"]
for i in 1:3
    velprofplt_lo = plot(fpxvec, ana_vels_fat_core[i, :], label="Analy sol finite")
    plot!(fpxvec, ana_vel_thin_core[i, :], label="Analy sol infinite")
    plot!(fpxvec, num_vels[i, :], label="Numer solution")
    vline!([CORERADIUS, CORERADIUS*5, CORERADIUS*10], label="Core radii (1, 5x, 10x)")
    title!("Velocity Profile of Lamb-Oseen Vortex\n$(axeslabel[i])-axis at z=$(ZPLANE)")
    xlabel!("x")
    ylabel!("y")
    ylims!(0, 0.6)
    display(velprofplt_lo)
end