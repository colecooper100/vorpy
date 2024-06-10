###### Import modules and local scrips ######
using CUDA
using StaticArrays

# environment_variables.jl sets the path varibles
# used to call other scripts
# pwd() returns the "present working directory". For
# this project, pwd() should return the path to the
# vorpy directory
include(string(pwd(), "/julia_src/environment_variables.jl"))
include(string(WEIGHTED_BIOT_SAVART_INTEGRATOR_METHODS, "/bimodal_integrator_polygonal_segments/polygonal_line_segmenter.jl"))


###### Set line segment and field point ######
# CuArray objects cannot be used to do elementwise
# operations. So rather than using CuArray objects
# for the input values, we will use SVector objects.
# (-1, 0, -2) (5, 2, 3.5)
VPP1 = SVector{3, Float32}(-1, 0, -2)  # (0, 0, 0)
VPP2 = SVector{3, Float32}(5, 2, 3.5)  # (5, 0, 0)
CUTOFF = Float32(5)
# Test cases when VPP1 = [0, 0, 0] and VPP2 = [5, 0, 0]:
# fp before p1 but part of path is inside cutoff: (-2, 2, 0)
# fp before p1 but path is outside cutoff: (-2, 4.75, 0)
# fp between p1 and p2 but all of path is inside cutoff: (2.5, 2, 0)
# fp between p1 and p2 but part of path is inside cutoff: (2.5, 4.9, 0)
# fp after p2 but part of path is inside cutoff: (7, 2, 0)
# fp after p2 but path is outside cutoff: (7, 4.7, 0)
FP = SVector{3, Float32}(0, 5, 2)
println("VPP1 = ", VPP1)  # DEBUG
println("VPP2 = ", VPP2)  # DEBUG
println("CUTOFF = ", CUTOFF)  # DEBUG
println("FP = ", FP)  # DEBUG


###### Run on the CPU ######
# The integrand always expects ell \in [0, seglen]
# biot_savart_integrand(ell, fp, vpp1, vpp2, crad1, crad2, circ1, circ2)
cpu_pathsegincutoff_rtn, cpu_pathsegoutcutoff_rtn = polygonal_line_segmenter(CUTOFF,
                                                                                FP,
                                                                                VPP1,
                                                                                VPP2)

println("pathsegincutoff_rtn_cpu = ", cpu_pathsegincutoff_rtn)  # DEBUG
# println("length(pathsegincutoff_rtn_cpu) = ", length(cpu_pathsegincutoff_rtn))  # DEBUG
println("pathsegoutcutoff_rtn_cpu = ", cpu_pathsegoutcutoff_rtn)  # DEBUG
# println("length(pathsegoutcutoff_rtn_cpu) = ", length(cpu_pathsegoutcutoff_rtn))  # DEBUG


###### Run on the GPU ######
# Make a wrapper to run functions on the GPU
function gpu_kernel(pathsegincutoff_rtn, pathsegoutcutoff_rtn,  # return values
                    cutoffrad, fp, vpp1, vpp2)  # input values
    
    rtn1, rtn2 = polygonal_line_segmenter(cutoffrad, fp, vpp1, vpp2)
    # I use this odd notation because if I try to do [1][1]
    # to access the boolean value, I get a "scalar indexing"
    # error. Basically, you can't index one value at a time
    # you need to index a slice of the array.
    # Another way this could be done is to index the first
    # element and then just broadcast the result to the
    # "whole" array (i.e., the array with only one element
    # but you act like you are assigning the same value to
    # every element).
    # Remember rtn1 will have the form (Bool, (SVector{3, Float32}, SVector{3, Float32}))
    # and rtn2 will have the form (1:(1:Bool, 2:(1:SVector{3, Float32}, 2:SVector{3, Float32})), 2:(1:Bool, 2:(1:SVector{3, Float32}, 2:SVector{3, Float32})))
    pathsegincutoff_rtn[1] .= rtn1[1]  # boolean value
    pathsegincutoff_rtn[2][1] .= rtn1[2][1]  # first vector
    pathsegincutoff_rtn[2][2] .= rtn1[2][2]  # second vector
    pathsegoutcutoff_rtn[1][1] .= rtn2[1][1]  # first elm boolean value
    pathsegoutcutoff_rtn[1][2][1] .= rtn2[1][2][1]  # first elm first vector
    pathsegoutcutoff_rtn[1][2][2] .= rtn2[1][2][2]  # first elm second vector
    pathsegoutcutoff_rtn[2][1] .= rtn2[2][1]  # second elm boolean value
    pathsegoutcutoff_rtn[2][2][1] .= rtn2[2][2][1]  # second elm first vector
    pathsegoutcutoff_rtn[2][2][2] .= rtn2[2][2][2]  # second elm second vector 

    return nothing
end

# Allocate memory on the GPU
# When returning values from the GPU, we need to
# to define variables that the GPU can modify
# in place (i.e., a mutable object), even when 
# returning a scalar. This is because the GPU's
# memory is on the device (i.e., separate from 
# the CPU's). This is why we use CuArray{Float32}(undef, 1)
# instead of Float32(undef, 1).
gpu_pathsegincutoff_rtn = (CuArray{Bool}(undef, 1), (CuArray{Float32}(undef, 3, 1), CuArray{Float32}(undef, 3, 1)))
gpu_pathsegoutcutoff_rtn = ((CuArray{Bool}(undef, 1), (CuArray{Float32}(undef, 3, 1), CuArray{Float32}(undef, 3, 1))), (CuArray{Bool}(undef, 1), (CuArray{Float32}(undef, 3, 1), CuArray{Float32}(undef, 3, 1))))

# Run the kernel
# @device_code_warntype
# @device_code_llvm
@device_code_warntype @cuda gpu_kernel(gpu_pathsegincutoff_rtn, gpu_pathsegoutcutoff_rtn,
                                        CUTOFF, FP, VPP1, VPP2)

println("gpu_pathsegincutoff_rtn: ", gpu_pathsegincutoff_rtn)  # DEBUG
println("gpu_pathsegoutcutoff_rtn: ", gpu_pathsegoutcutoff_rtn)  # DEBUG




###### Plot result ######
using Plots

vec1 = FP .- VPP1
ptan = (VPP2 .- VPP1) ./ norm(VPP2 .- VPP1)
vec1tan = dot(vec1, ptan)
vec1perp = norm(vec1 .- vec1tan * ptan)

plt = plot([VPP1[1], VPP2[1]], [VPP1[2], VPP2[2]], [VPP1[3], VPP2[3]], label="Line segment", linewidth=2, markershape=:circle, linecolor=:blue, markercolor=:blue)
scatter!([FP[1]], [FP[2]], [FP[3]], label="Field point", markershape=:x, markercolor=:black)
# plot!([VPP1[1], VPP1[1]+vec1tan], [VPP1[2], VPP1[2]], [VPP1[3], VPP1[3]], label="vec1 adjacent", markershape=:none, linecolor=:red)
# plot!([VPP1[1]+vec1tan, VPP1[1]+vec1tan], [VPP1[2], FP[2]], [VPP1[3], FP[3]], label="vec1 opposite", markershape=:none, linecolor=:red)

if Array(gpu_pathsegincutoff_rtn[1])[1]
    incutpts = [Array(gpu_pathsegincutoff_rtn[2][1]) Array(gpu_pathsegincutoff_rtn[2][2])]
    plot!(incutpts[1, :], incutpts[2, :], incutpts[3, :], label="Inside cutoff", linewidth=2, markershape=:none, linecolor=:green, markercolor=:green)
end
if Array(gpu_pathsegoutcutoff_rtn[1][1])[1]
    outcutpts1 = [Array(gpu_pathsegoutcutoff_rtn[1][2][1]) Array(gpu_pathsegoutcutoff_rtn[1][2][2])]
    plot!(outcutpts1[1, :], outcutpts1[2, :], outcutpts1[3, :], label="Outside cutoff", linewidth=2, markershape=:none, linecolor=:red, markercolor=:red)
end
if Array(gpu_pathsegoutcutoff_rtn[2][1])[1]
    outcutpts2 = [Array(gpu_pathsegoutcutoff_rtn[2][2][1]) Array(gpu_pathsegoutcutoff_rtn[2][2][2])]
    plot!(outcutpts2[1, :], outcutpts2[2, :], outcutpts2[3, :], label="Outside cutoff", linewidth=2, markershape=:none, linecolor=:yellow, markercolor=:yellow)
end

plot!(plt, camera=(20, 85), xlabel="x", ylabel="y", zlabel="z")

display(plt)