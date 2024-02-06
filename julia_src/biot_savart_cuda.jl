using CUDA
using StaticArrays
using LinearAlgebra


function slice3(arr, i)
    return @inbounds SVector{3, Float32}(arr[1, i], arr[2, i], arr[3, i])
end

function uniform_trapezoidal_rule(fn, a, b, numsteps=100)
    du = (b - a) ./ numsteps
    rtn_vel = @MVector zeros(3)
    rtn_vel .+= 0.5 * (fn(a) + fn(b))
    for i in 1:numsteps-1
        rtn_vel .+= fn(a + i*du)
    end
    return rtn_vel .* du
end


#=
- [x] Pass all the arguments to the kernel and
    print some info from the kernel.
    - [x] returnvelocities
    - [x] fieldpointbatch
    - [x] vortexpath
    - [x] vortexcore
    - [x] weightfn
    - [!] integrator
    - [-] circulation (*see log for note*)
- [x] Use the global index to get the field point
    from the batch and store it in rtnvelocities
- [ ] Check if functions from LinearAlgebra work
    on the GPU
    - [?] norm
    - [?] cross
- [ ] Loop through the vortex path points
    - [ ] Get the segment vector
    - [ ] Get the tangent vector
    - [ ] Define the path function
    - [ ] Define the xi function
    - [ ] Define the core function
    - [ ] Define the integrand function
    - [ ] Integrate the integrand function
    - [ ] Add the result to the return velocities
=#
#=
**Log**
Passing the integrator to the kernel
- I was having issues trying to pass the integrator
    to the kernel. I think this is because the function
    needs to be a type that can be used on the GPU (as in,
    a CUDA callable function). Rather than passing a
    function, I will hard code the integrator into the
    kernel. (Functions can still be written outside of
    the kernel and called from within the kernel, you
    just can't pass them as arguments to the kernel.)
Passing circulation as a keyword argument
- I wasn't able to pass circulation as a keyword
    argument to the kernel. I think this is because
    keyword arguments are not supported in CUDA kernels.
    So, I am passing circulation as a positional argument.
LinearAlgebra functions on the GPU
- I was able to use the norm function from LinearAlgebra
    on the GPU, but I'm still not sure if it is actually
    running on the GPU or if it is making a call to the
    CPU.
=#
function weighted_biot_savart_kernel(
    returnvelocities,
    fieldpointbatch,
    vortexpath,
    vortexcore,
    weightfn,
    circulation=1)

    # Compute the global index of the thread
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    # Get the field point from the batch
    @inbounds fieldpoint = slice3(fieldpointbatch, idx)
    
                                     # DEBUG
    # @inbounds returnvelocities[:, idx] .= norm(fieldpoint) .+ cross(fieldpoint, fieldpoint)
    # CUDA.@cuprint(uniform_trapezoidal_rule())

    # Step through each segment of the vortex path
    rtn_vel = @MVector zeros(3)
    for i in 1:(size(vortexpath, 2) -1)
        vseg = slice3(vortexpath, i+1) .- slice3(vortexpath, i)
        vsegmag = norm(vseg)
        vsegtan = vseg ./ vsegmag
        path(ell) = slice3(vortexpath, i) .+ ell .* vsegtan
        xi(ell) = fieldpoint .- path(ell)
        # core(ell) = vortexcore[i] .+ ell .* (vortexcore[i+1] .- vortexcore[i]) / vsegmag
        function integrand(ell, params...)
            xiell = xi(ell)
            xiellmag = norm(xiell)
            # c = core(ell)
            dir = cross(vsegtan, xiell)
            # w = weightfn(xiellmag / c)
            return (1i32 / xiellmag^3) .* dir
        end
        rtn_vel .+= vsegtan .+ uniform_trapezoidal_rule(integrand, 0, vsegmag)  # DEBUG
        # rtn_vel .+= circulation .* integrator(integrand, 0, vsegmag) ./ (4*pi)
    end

    returnvelocities[:, idx] .= rtn_vel
    return nothing
end



#============ Straight Vortex Test Case ============#
#=
**Straight Vortex Test Case**
This is the start of a unit test of the
weighted_biot_savart_kernel function.

Let a infinitely long straight vortex be aligned with
the x-axis. We will evaluate the velocity at points
along the y-axis. 
=#
using Plots

# Define the vortex path and core
vp = CuArray{Float32}([[-1000, 0, 0] [1000, 0, 0]])  # [0 1; 0 0; 0 0] same same
vc = CuArray{Float32}([0.1, 0.1])

# Define the field points
y = zeros(Float32, 10)
y .= collect(0:2:18)  # end point is included in range
y[1] = 1e-3  # avoid divide by zero
fp = zeros(Float32, 3, 10)
fp[2, :] = y
fp = CuArray{Float32}(fp)

# Create the return velocities array
# CuArray is mutable!!!
rtnvelocities = CuArray{Float32}(undef, 3, 10)
print("rtnvelocities (before):")
display(rtnvelocities)

# Call the kernel
@cuda blocks=2 threads=3 weighted_biot_savart_kernel(
    rtnvelocities,
    fp,
    vp,
    vc,
    x->1,
    2)

# CUDA.CUDA.@profile external=true @cuda blocks=2 threads=3 weighted_biot_savart_kernel(
#     rtnvelocities,
#     fp,
#     vp,
#     vc,
#     x->1,
#     2)

print("rtnvelocities (after):")
display(rtnvelocities)
# # println("vel = ", vel)
# # println("vel_true = ", vel_true)
# println("Error (straight vortex): ", norm(vel - vel_true))

# plt = plot(y, vel, markershape=:x, label="Numerical")
# plot!(y, vel_true, label="Analytical")
# display(plt)