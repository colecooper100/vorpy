###### Set the path to the `julia_fns` directory ######
# julia_env.jl sets global varibles used by
# other scrips in the `julia_fns` directory.
# This is an absolute path so it will be
# different on different systems. This could
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
include(string(JULIA_FNS, "/weighted_biot_savart_solver_cpu.jl"))


###### Make a vortex ######
NUMSEGS = 100
phi = range(0, stop=2*pi, length=NUMSEGS + 1)
println("length(phi) = ", length(phi))  # DEBUG
r = @. 1 + 0.25 * cos(6 * phi)
vppsx = @. r * cos(phi)
vppsy = @. r * sin(phi)
vpps = zeros(Float32, 3, NUMSEGS + 1)
vpps[1, :] = vppsx
vpps[2, :] = vppsy
# Make the last point and first point the same
# so the vortex is closed
vpps[:, end] .= vpps[:, 1]

crads = ones(Float32, NUMSEGS + 1)
circs = ones(Float32, NUMSEGS + 1)

# Plot vortex
plt = plot(vpps[1, :], vpps[2, :], label="vortex")
plot!(plt, title="Vortex in the xy-plane", xlabel="x", ylabel="y")


###### Make field points ######
fps = copy(vpps)
# fps = zeros(Float32, 3, 3)
# fps[:, 1] .= vpps[:, 1]
# fps[:, 2] .= vpps[:, 9]
# fps[:, 3] .= vpps[:, 22]

# Plot field points
scatter!(plt, fps[1, :], fps[2, :], label="field points")


###### Calculate the velocity at the field points ######
vels = weighted_biot_savart_solver_cpu(fps, vpps, crads, circs)


