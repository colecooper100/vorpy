###### Import modules and local scrips ######
using CUDA
using StaticArrays

# environment_variables.jl sets the path varibles
# used to call other scripts
# pwd() returns the "present working directory". For
# this project, pwd() should return the path to the
# vorpy directory
include(string(pwd(), "/julia_fns/src/environment_variables.jl"))
include(string(WEIGHTED_BIOT_SAVART_INTEGRAND, "/weighed_biot_savart_integrand.jl"))


###### Set the model of the vortex ######
include(string(VORTEX_MODELS, "/piecewise_linear_vortex_segment_model.jl"))
function vortex_model(ell, vpp1, vpp2, crad1, crad2, circ1, circ2)
    return piecewise_linear_vortex_segment_model(ell, vpp1, vpp2, crad1, crad2, circ1, circ2)
end


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
vpps = zeros(3, NUMSEGS + 1)
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
sfp = SVector{3, Float32}(FP)
svpp1 = SVector{3, Float32}(vpps[:, 1])
svpp2 = SVector{3, Float32}(vpps[:, 2])


###### Run on the CPU ######
# The integrand always expects ell \in [0, seglen]
# biot_savart_integrand(ell, fp, vpp1, vpp2, crad1, crad2, circ1, circ2)
cpu_rnt = weighted_biot_savart_integrand(ELL,
                                            sfp,
                                            svpp1,
                                            svpp2,
                                            crads[1],
                                            crads[2],
                                            circs[1],
                                            circs[2])

println("cpu_rnt = ", cpu_rnt)  # DEBUG


###### Run on the GPU ######
# Make a wrapper to run functions on the GPU
function gpu_kernel(integrand_vec_rtn, ell_rtn, endofseg,  # return values
                        ell, fp, vpp1, vpp2, crad1, crad2, circ1, circ2)  # input values
    
    rtn = weighted_biot_savart_integrand(ell, fp, vpp1, vpp2, crad1, crad2, circ1, circ2)
    
    integrand_vec_rtn .= rtn[1]
    ell_rtn .= rtn[2]
    endofseg .= rtn[3]

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
gpu_integrand_vec_rtn = CuArray{Float32}(undef, 3)
gpu_ell_rtn = CuArray{Float32}(undef, 1)
gpu_endofseg = CuArray{Bool}(undef, 1)

# Run the kernel
# @device_code_warntype
# @device_code_llvm
@device_code_warntype @cuda gpu_kernel(gpu_integrand_vec_rtn, gpu_ell_rtn, gpu_endofseg,
                                        ELL, sfp, svpp1, svpp2, crads[1], crads[2], circs[1], circs[2])

println("GPU return: ", gpu_integrand_vec_rtn, ", ", gpu_ell_rtn, ", ", gpu_endofseg)