###### Import modules and local scrips ######
using CUDA
using StaticArrays

# environment_variables.jl sets the path varibles
# used to call other scripts
# pwd() returns the "present working directory". For
# this project, pwd() should return the path to the
# vorpy directory
include(string(pwd(), "/julia_src/environment_variables.jl"))
include(string(WEIGHTED_BIOT_SAVART_SOLVER_ONE_FIELD_POINT, "/weighted_biot_savart_solver_one_field_point.jl"))


###### Test and vortex parameters ######
ENDPOINTS = (-100, 100)
NUMSEGS = 10
CORERADIUS = Float32(5)
CIRCULATION = Float32(10.0)
FP = SVector{3, Float32}(0, 2, 0)


###### Make the vortex ######
# Make the vortex path
# Align the vortex with the z-axis
vpps = zeros(Float32, 3, NUMSEGS + 1)
vpps[3, :] .= range(ENDPOINTS[1], stop=ENDPOINTS[2], length=NUMSEGS + 1)
println("typeof(vpps) = ", typeof(vpps))  # DEBUG
println("size(vpps) = ", size(vpps))  # DEBUG

# Define the core radii
# Eventually I would like to allow the core radius
# to vary along the path, but for now, we will just
# use a constant radius.
crads = ones(Float32, NUMSEGS + 1) .* CORERADIUS
println("typeof(cor_rads) = ", typeof(crads))  # DEBUG
println("size(cor_rads) = ", size(crads))  # DEBUG

# Define the circulations
# Eventually I would like to allow the circulations
# to vary along the path, but for now, we will just
# use a constant circulation.
circs = ones(Float32, NUMSEGS + 1) .* CIRCULATION
println("typeof(circs) = ", typeof(circs))  # DEBUG
println("size(circs) = ", size(circs))  # DEBUG


###### Run on the CPU ######
# piecewise_linear_vortex_segment_model(ell, vpp1, vpp2, vcr1, vcr2, cir1, cir2, params=nothing)
cpu_rtn = weighted_biot_savart_for_one_field_point(FP,
                                                    vpps,
                                                    crads,
                                                    circs)

println("typeof(cpu_rtn) = ", typeof(cpu_rtn))  # DEBUG
println("CPU return: ", cpu_rtn)


###### Run on the GPU ######
# Make a wrapper to run functions on the GPU.
# Kernel functions needs to return `nothing`
function gpu_kernel(vol_rtn,  # return values
                    fp, vpps, crads, circs)  # input values
    
    # piecewise_linear_vortex_segment_model(ell, vpp1, vpp2, vcr1, vcr2, cir1, cir2, params=nothing)
    vol_rtn .= weighted_biot_savart_for_one_field_point(fp, vpps, crads, circs)

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
gpu_vol_rtn = CuArray{Float32}(undef, 3)
gpu_vpps = CuArray{Float32}(vpps)
gpu_crads = CuArray{Float32}(crads)
gpu_circs = CuArray{Float32}(circs)

# Run the kernel
# @device_code_warntype
# @device_code_llvm
@device_code_warntype @cuda gpu_kernel(gpu_vol_rtn,
                                        FP, gpu_vpps, gpu_crads, gpu_circs)

println("GPU return: ", gpu_vol_rtn)
