###### Import modules and local scrips ######
using CUDA
using StaticArrays

# environment_variables.jl sets the path varibles
# used to call other scripts
# pwd() returns the "present working directory". For
# this project, pwd() should return the path to the
# vorpy directory
include(string(pwd(), "/julia_fns/src/environment_variables.jl"))
include(string(VORTEX_MODELS, "/piecewise_linear_vortex_segment_model.jl"))


###### Test and vortex parameters ######
ENDPOINTS = (-100, 100)
NUMSEGS = 1
CORERADIUS = Float32(5)
CIRCULATION = Float32(10.0)
ELL = Float32(50)
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


###### Make the data passible to function ######
# CuArray objects cannot be used to do elementwise
# operations. So rather than using CuArray objects
# for the input values, we will use SVector objects.
svpp1 = SVector{3, Float32}(vpps[:, 1])
svpp2 = SVector{3, Float32}(vpps[:, 2])


###### Run on the CPU ######
# piecewise_linear_vortex_segment_model(ell, vpp1, vpp2, vcr1, vcr2, cir1, cir2, params=nothing)
cpu_rtn = piecewise_linear_vortex_segment_model(ELL,
                                                svpp1,
                                                svpp2,
                                                crads[1],
                                                crads[2],
                                                circs[1],
                                                circs[2])

println("typeof(cpu_rtn) = ", typeof(cpu_rtn))  # DEBUG
println("CPU return: ", cpu_rtn)


###### Run on the GPU ######
# Make a wrapper to run functions on the GPU.
# Kernel functions needs to return `nothing`
function gpu_kernel(vpp_rtn, vtan, crad_rtn, circ_rtn, ell_rtn, endofseg,  # return values
                     ell, vpp1, vpp2, crad1, crad2, circ1, circ2)  # input values
    
    # piecewise_linear_vortex_segment_model(ell, vpp1, vpp2, vcr1, vcr2, cir1, cir2, params=nothing)
    rtn = piecewise_linear_vortex_segment_model(ell, vpp1, vpp2, crad1, crad2, circ1, circ2)

    vpp_rtn .= rtn[1]
    vtan .= rtn[2]
    crad_rtn .= rtn[3]
    circ_rtn .= rtn[4]
    ell_rtn .= rtn[5]
    endofseg .= rtn[6]

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
gpu_vpp_rtn = CuArray{Float32}(undef, 3)
gpu_vtan = CuArray{Float32}(undef, 3)
gpu_crad_rtn = CuArray{Float32}(undef, 1)
gpu_circ_rtn = CuArray{Float32}(undef, 1)
gpu_ell_rtn = CuArray{Float32}(undef, 1)
gpu_endofseg = CuArray{Bool}(undef, 1)

# Run the kernel
# @device_code_warntype
# @device_code_llvm
@device_code_warntype @cuda gpu_kernel(gpu_vpp_rtn, gpu_vtan, gpu_crad_rtn, gpu_circ_rtn, gpu_ell_rtn, gpu_endofseg,
                                        ELL, svpp1, svpp2, crads[1], crads[2], circs[1], circs[2])

println("GPU return: ", gpu_vpp_rtn, ", ", gpu_vtan, ", ", gpu_crad_rtn, ", ", gpu_circ_rtn, ", ", gpu_ell_rtn, ", ", gpu_endofseg)