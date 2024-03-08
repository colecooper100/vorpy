using CUDA
using Plots
using Statistics
using BenchmarkTools


#===============================================
This should be a diagonstic tool. That is, it 
tells you when the BS stripts aren't working and
provides some infromation about where the problem
might be coming from.

This is not made to play with the vortex and 
do research related things.
===============================================#


################################################
# Import Biot-Savart code
include("vortex_models.jl")
# include("biot_savart_integrator.jl")
# include("biot_savart_cpu.jl")


################################################
#========== Straight Vortex Test Case ==========


We define a Lamb-Oseen vortex aligned with the
z-axis.

If our coordinate system has +z running from
left-to-right, then +x is into the page and +y
is bottom-to-top. We compute the flow velocity
at points along the +x-axis at the origin, i.e.,
$\vec r = (x, 0, 0)$.
===============================================#

# Set problem parameters
VDOMAIN = [-1000, 1000]  # Vortex path domain
FPDOMAIN = [0, 20]  # Field points domain
NUMVPSEGS = 3  # Number of vortex path segments
NUMFP = 5  # Number of field points
ERRTOL = 1f-4  # Error tolerance for results comparison
println("VDOMAIN = ", VDOMAIN)
println("FPDOMAIN = ", FPDOMAIN)
println("NUMVPSEGS = ", NUMVPSEGS)
println("NUMFP = ", NUMFP)
println("ERRTOL = ", ERRTOL)

# Define the vortex
## Path points
vpps = zeros(Float32, 3, NUMVPSEGS + 1)
vpps[1, :] .= range(VDOMAIN[1], stop=VDOMAIN[2], length=NUMVPSEGS + 1)
## Core diameters
vcrs = Float32.(collect(axes(vpps, 2)))  # ones(Float32, NUMVPSEGS + 1)
## Circulation
cirs = Float32.(collect(axes(vpps, 2)))  # ones(Float32, NUMVPSEGS + 1)

# Define the field points
x = zeros(Float32, NUMFP)
x .= range(FPDOMAIN[1], FPDOMAIN[2], length=NUMFP)  # end point is included in range
if x[1] == 0 
    x[1] = 1e-3  # avoid divide by zero
end
fps = zeros(Float32, 3, NUMFP)
fps[1, :] .= x
# println("fps = ")  # DEBUG
# display(fps)  # DEBUG


################## Test vortex model ##################

#==========================================
The vortex model values at ell=0 and ell=end
are known so the error should be small. 

The tests for ell=0 and ell=half the domain
are mostly gusses, so the error may be larger.
==========================================#
spandomain = VDOMAIN[2] - VDOMAIN[1]
ellvec = Float32.([0, 500, spandomain/2, spandomain])
vppans = [vpps[:, 1], [-500, 0, 0], [0, 0, 0], vpps[:, end]]
vtanans = [[1e0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]
vcrans = [vcrs[1], 1.5, 2, vcrs[end]]
cirans = [vcrs[1], 1.5, 2, vcrs[end]]

println("################ CPU ################")
for i in eachindex(ellvec)
    # println("elm = ", elm)  # DEBUG
    # @btime piecewise_linear_path(ellvec[$i], $vpps, $vcrs, $cirs)
    vpp, vtan, vcr, cir = piecewise_linear_path(ellvec[i], vpps, vcrs, cirs)
    # println("vpp = ", vpp)  # DEBUG
    vpperr = abs.(vpp .- vppans[i])
    println("vpp: ", vpperr .> ERRTOL, ", Error", vpperr)
    # println("vtan = ", vtan)  # DEBUG
    vtanerr = abs.(vtan .- vtanans[i])
    println("vtan: ", vtanerr .> ERRTOL, ", Error", vtanerr)
    # println("vcr = ", vcr)  # DEBUG
    # println("vcrans = ", vcrans[i])  # DEBUG
    vcrerr = abs(vcr .- vcrans[i])
    println("vcr: ", vcrerr .> ERRTOL, ", Error ", vcrerr)
    # println("cir = ", cir)  # DEBUG
    # println("cirans = ", cirans[i])  # DEBUG
    cirerr = abs(cir .- cirans[i])
    println("cir: ", cirerr .> ERRTOL, ", Error ", cirerr)
    println("-----------------------------")
end

println("################ GPU ################")





# @cuda launch=false kernel_function(view(cuellvec, 1), cuvpps, cuvcrs, cucirs)

# for i in eachindex(ellvec)
#     vpp, vtan, vcr, cir = piecewise_linear_path(ellvec[i], vpps, vcrs, cirs)
#     vpperr = abs.(vpp .- vppans[i])
#     println("vpp: ", vpperr .> ERRTOL, ", Error", vpperr)
#     vtanerr = abs.(vtan .- vtanans[i])
#     println("vtan: ", vtanerr .> ERRTOL, ", Error", vtanerr)
#     vcrerr = abs(vcr .- vcrans[i])
#     println("vcr: ", vcrerr .> ERRTOL, ", Error ", vcrerr)
#     cirerr = abs(cir .- cirans[i])
#     println("cir: ", cirerr .> ERRTOL, ", Error ", cirerr)
#     println("-----------------------------")
# end
 

# ################ Results ################

# # Test run the precompiled kernel
# vel_num = bs_solve(fps, vps, vcrs, cirs)

# # Analytical solution (infinitesimally thin vortex)
# vel_true_nocore = Float32.(1 ./ (2 .* pi .* y))

# # Analytical solution (finite core)
# vel_true_core = Float32.(1 ./ (2 .* pi .* y) .* (1 .- exp.(-y.^2 ./ (2 * vcrs[1]^2))))

# # println("vel_true = ", vel_true)
# println("Total L2 error (straight vortex): ", norm(vel_num[3, :] - vel_true_core))
# println("Avarage L2 error (straight vortex): ", sqrt(mean((vel_num[3, :] - vel_true_core).^2)))

# println("vel_true_core = ")  # DEBUG
# display(vel_true_core)  # DEBUG
# println("vel_num = ")  # DEBUG
# display(vel_num[3, :])  # DEBUG

# stride = 1
# pltvelmag = plot(y[1:stride:end], vel_num[3, 1:stride:end], markershape=:o, label="Numerical")
# plot!(pltvelmag, y[1:stride:end], vel_true_nocore[1:stride:end], markershape=:x, label="Analytical (No Core)")
# plot!(pltvelmag, y[1:stride:end], vel_true_core[1:stride:end], markershape=:x, label="Analytical (Core)")
# xlabel!("y")
# ylabel!("Velocity")
# # xlims!(0, 1)
# ylims!(0, 0.1)
# title!(pltvelmag, "Straight Vortex Test Case,\nVelocity at every $(stride)th Field Point")
# display(pltvelmag)

# # plterror = plot(y[1:stride:end], abs.(vel_num[3, 1:stride:end] .- vel_true[1:stride:end]), markershape=:x, label="Error")
# # title!(plterror, "Straight Vortex Test Case - Absolute Error")
# # xlabel!("y")
# # ylabel!("Absolute Error")
# # ylims!(0, .1)

# # display(plterror)