###### Import modules and local scrips ######
using CUDA
using StaticArrays

# environment_variables.jl sets the path varibles
# used to call other scripts
# pwd() returns the "present working directory". For
# this project, pwd() should return the path to the
# vorpy directory
include(string(pwd(), "/julia_src/environment_variables.jl"))
include(string(WEIGHTED_BIOT_SAVART_INTEGRAND, "/bernstein_polynomial_weight.jl"))


###### Test parameters ######
DELTA = Float32(0.5)


###### Run on the CPU ######
cpu_rtn = bernstein_polynomial_weight(DELTA)
println("CPU return: ", cpu_rtn)


###### Run on the GPU ######
function gpu_kernel(rtn,  # return values
                    delta)  # input values

    rtn .= bernstein_polynomial_weight(delta)

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
gpu_rtn = CuArray{Float32}(undef, 1)

# Run the kernel
# @device_code_warntype
# @device_code_llvm
@device_code_warntype @cuda gpu_kernel(gpu_rtn, DELTA)
println("GPU return: ", gpu_rtn)