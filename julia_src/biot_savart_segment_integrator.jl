using StaticArrays: SVector
using LinearAlgebra: norm, cross

include("vortex_segment_models.jl")
include("bernstein_polynomial_weight.jl")

# Set vortex segment model
vortex_segment_model(ell, vpp1, vpp2, vcr1, vcr2, cir1, cir2) = piecewise_linear_vortex(ell, vpp1, vpp2, vcr1, vcr2, cir1, cir2)

# Set weight function
weight_function(delta) = bernstein_polynomial_weight(delta)

# The integrand always expects ell \in [0, seglen]
function bs_integrand_segment(ell, fp, vpp1, vpp2, vcr1, vcr2, cir1, cir2)
    vpp, vtan, vcr, cir, rtnell, endofseg = vortex_segment_model(ell, vpp1, vpp2, vcr1, vcr2, cir1, cir2)
    xi = fp .- vpp
    ximag = norm(xi)
    dir = cross(vtan, xi)
    weight = weight_function(ximag / vcr)
    return (weight * cir / ximag^3) .* dir, rtnell, endofseg
end

# Integrate along a given path segment
# F(x) = \int_a^b f(x) dx \approx \sum_{i=1}^{N} (f(x_{i-1}) + f(x_i))/2 * \Delta x_i
# where \Delta x_i = x_i - x_{i-1}
function bs_nonuniform_trapezoidal_rule_segment(stepsize, fp, vpp1, vpp2, vcr1, vcr2, cir1, cir2)
    # Initialize variables used in the loop
    # totell = Float32(0)
    sol = SVector{3, Float32}(0, 0, 0)
    eval_bs_integrand, ell, endofseg = bs_integrand_segment(Float32(0), fp, vpp1, vpp2, vcr1, vcr2, cir1, cir2)
    while !endofseg
        prev_ell = ell
        prev_eval_bs_integrand = eval_bs_integrand
        eval_bs_integrand, ell, endofseg = bs_integrand_segment(prev_ell + stepsize, fp, vpp1, vpp2, vcr1, vcr2, cir1, cir2)
        # totell += rtnell - ell
        sol = sol .+ (prev_eval_bs_integrand .+ eval_bs_integrand) * (ell - prev_ell) / 2
    end
    return sol / (4 * Float32(pi))
end



# ################### DEBUGGING ###################
# using CUDA
# using BenchmarkTools

# # Get a path segment
# function get_segment(vpps, vcrs, cirs, indx)
#     # Get starting point of segment
#     @inbounds vpp1 = SVector{3, Float32}(
#         vpps[1, indx],
#         vpps[2, indx],
#         vpps[3, indx])

#     # Get the ending point of segment
#     @inbounds vpp2 = SVector{3, Float32}(
#         vpps[1, indx+1],
#         vpps[2, indx+1],
#         vpps[3, indx+1])

#     return vpp1, vpp2, vcrs[indx], vcrs[indx+1], cirs[indx], cirs[indx+1]
# end

# #==============================================
# -----------Straight Vortex Test Case-----------
# We define a Lamb-Oseen vortex aligned with the
# z-axis.

# If our coordinate system has +z running from
# left-to-right, then +x is into the page and +y
# runs from down-to-up. We compute the flow
# velocity at points along the +x-axis at the
# origin, i.e., $\vec r = (x, 0, 0)$.
# ==============================================#
# # Set problem parameters
# VDOMAIN = [-1000, 1000]  # Vortex path domain
# FPDOMAIN = [0, 20]  # Field points domain
# NUMVPSEGS = 3  # Number of vortex path segments
# NUMFP = 5  # Number of field points
# println("VDOMAIN = ", VDOMAIN)
# println("FPDOMAIN = ", FPDOMAIN)
# println("NUMVPSEGS = ", NUMVPSEGS)
# println("NUMFP = ", NUMFP)

# # Define the vortex
# vpps = zeros(Float32, 3, NUMVPSEGS + 1)  # Path points
# vpps[3, :] .= range(VDOMAIN[1], stop=VDOMAIN[2], length=NUMVPSEGS + 1)
# # Core diameters
# vcrs = ones(Float32, NUMVPSEGS + 1) # Float32.(collect(axes(vpps, 2)))
# # Circulations
# cirs = ones(Float32, NUMVPSEGS + 1)  # Float32.(collect(axes(vpps, 2)))

# # Define the field points
# x = zeros(Float32, NUMFP)
# x .= range(FPDOMAIN[1], FPDOMAIN[2], length=NUMFP)  # end point is included in range
# if x[1] == 0 
#     x[1] = 1e-3  # avoid divide by zero
# end
# fps = zeros(Float32, 3, NUMFP)
# fps[1, :] .= x



# ###### Run on the CPU ######
# # vpp1, vpp2, vcr1, vcr2, cir1, cir2 = get_segment(vpps, vcrs, cirs, UInt32(2))
# # for i in 1:5
# #     rtncpu = bs_nonuniform_trapezoidal_rule_segment(Float32(1), fps[:, i], vpp1, vpp2, vcr1, vcr2, cir1, cir2)
# #     println("CPU: ", rtncpu)
# # end



# ###### Run on the GPU ######
# # Make a wrapper to run functions on the GPU
# function gpu_wrapper(rtn, fps, vpps, vcrs, cirs)
#     for fpindx in 1:5
#         # fpindx = UInt32(1)
#         fp = SVector{3, Float32}(fps[1, fpindx], fps[2, fpindx], fps[3, fpindx])
#         vpp1, vpp2, vcr1, vcr2, cir1, cir2 = get_segment(vpps, vcrs, cirs, UInt32(2))
#         sol = bs_nonuniform_trapezoidal_rule_segment(Float32(1), fp, vpp1, vpp2, vcr1, vcr2, cir1, cir2)
#         rtn[:, fpindx] .= sol
#     end
#     return nothing
# end

# # Allocate memory on the GPU
# cufps = CuArray(fps)
# cuvpps = CuArray(vpps)
# cuvcrs = CuArray(vcrs)
# cucirs = CuArray(cirs)
# curtn = CuArray{Float32}(undef, 3, 5)



# # @device_code_warntype
# # @device_code_llvm
# # @cuda gpu_wrapper(curtn, cufps, cuvpps, cuvcrs, cucirs)
# # println("curtn = ")  # DEBUG
# # display(curtn)  # DEBUG

# kern = @cuda launch=false gpu_wrapper(curtn, cufps, cuvpps, cuvcrs, cucirs)
# println("Max threads: ", CUDA.maxthreads(kern))
# println("Register usage: ", CUDA.registers(kern))
# println("Memory usage: ", CUDA.memory(kern))




