using CUDA
using StaticArrays
using LinearAlgebra
using BenchmarkTools



################ Device functions ################

#=
I don't think you can define nested functions with
CUDA.jl, so, the functions used in the kernel are
defined outside of it here.

Note: The code doesn't have to be physically outside
of the kernel, you just can't assume you have access
to any variables other than what is passed to the
function and those defined inside of it. 
=#

#==================================== 
--------Vortex Path Model--------
Input:
    - vpps: vortex path points
    - sidx: segment index
    - ell \in [0, 1]: arc length
Output:
    - vortex path point at ell
    - unit tangent vector at ell

Note: we rescaled the arc length
    variable so the domain of ell is
    [0, 1] for all segments. This
    way we don't need to know the
    actual length of the vortex path
    or each segment.
====================================#
# Vortex path model: points are connected
# by straight lines.
function vpathmodel(vorpps, segidx, ell)
    @inbounds pnt1 = SVector{3, Float32}(
        vorpps[1, segidx],
        vorpps[2, segidx],
        vorpps[3, segidx])

    @inbounds seg = SVector{3, Float32}(
        vorpps[1, segidx+1] - pnt1[1],
        vorpps[2, segidx+1] - pnt1[2],
        vorpps[3, segidx+1] - pnt1[3])

    return pnt1 .+ (ell .* seg), seg ./ norm(seg)
end

function xi(fp, vorpps, segidx, ell)
    vpmodel = vpathmodel(vorpps, segidx, ell)
    @inbounds return fp .- vpmodel[1], vpmodel[2]
end

# Core model simple linear interpolation
function vcoremodel(corerads, segidx, ell)
    @inbounds return corerads[segidx] + ell * (corerads[segidx+1] - corerads[segidx])
end

# Circulation model simple linear interpolation
function vcircmodel(circs, segidx, ell)
    @inbounds return circs[segidx] + ell * (circs[segidx+UInt32(1)] - circs[segidx])
end

# Define the weight function
# x \in [0, 1]
function bsweightmodel(x)
    return Float32(1)
end

#====================================
--------BS integrand function--------
We will integrate the Biot-Savart
law from 0 to some cutoff radius.
After that, we will use the analytical
solution for the velocity field of a
infinitesimally thin vortex filament.
- At what cutoff radius does the
    solution of the BS law converge
    to that of the infinitesimally
    thin vortex filament?
====================================#
# Define the integrand function
function bsintegrand(fp, vorpps, corerads, circs, segidx, ell)
    xi_vptan = xi(fp, vorpps, segidx, ell)
    @inbounds xiellmag = norm(xi_vptan[1])
    corell = vcoremodel(corerads, segidx, ell)
    circell = vcircmodel(circs, segidx, ell)
    @inbounds direll = cross(xi_vptan[2], xi_vptan[1])
    weightell = bsweightmodel(xiellmag / corell)
    return (weightell * circell / xiellmag^3) .* direll
end

# Define BS solver
function bs_uniform_trapezoidal_rule(numsteps, fp, vorpps, corerads, circs, segidx)
    stepsize = Float32(1) / numsteps
    sol = bsintegrand(fp, vorpps, corerads, circs, segidx, Float32(0))
    sol = sol .+ bsintegrand(fp, vorpps, corerads, circs, segidx, Float32(1))
    sol = sol .* Float32(0.5)

    # Start stepindex at 2 because we already did
    # the first step and use 'less-than' becase we
    # already did the last step
    stepindex = UInt32(2)
    while stepindex < numsteps
        sol = sol .+ bsintegrand(fp, vorpps, corerads, circs, segidx, stepindex * stepsize)
        stepindex += UInt32(1)
    end

    return sol .* (stepsize / (4 * pi))
end


################ Biot-Savart kernel ################

#=
- [x] Pass all the arguments to the kernel and
    print some info from the kernel.
    - [x] returnvelocities
    - [x] fieldpointbatch
    - [x] vortexpath
    - [x] vortexcore
    - [x] weightfn
    - [!] integrator (*see log for note*)
    - [x] circulation (*see log for note*)
- [x] Use the global index to get the field point
    from the batch and store it in rtnvelocities
- [x] Check if functions from LinearAlgebra work
    on the GPU
    - [x] norm (probably? I assume they are running on the GPU)
    - [x] cross
- [x] Loop through the vortex path points
    - [x] Get the segment vector
    - [x] Get the tangent vector
    - [x] Define the path function
    - [x] Define the xi function
    - [x] Define the core function
    - [x] Define the integrand function
    - [x] Integrate the integrand function
    - [x] Add the result to the return velocities
=#
#=
**Design decisions**
Passing the integrator to the kernel
- I was having issues trying to pass the integrator
    to the kernel. I think this is because the function
    needs to be a type that can be used on the GPU (as in,
    a CUDA callable function). Rather than passing a
    function, I will hard code the integrator into the
    kernel. (Functions can still be written outside of
    the kernel and called from within the kernel, you
    just can't pass them as arguments to the kernel.)

Not passing circulation as a keyword argument
- I wasn't able to pass circulation as a keyword
    argument to the kernel. I think this is because
    keyword arguments are not supported in CUDA kernels.
    So, I am passing circulation as a positional argument.
=#
function weighted_biot_savart_kernel(
    returnvelocities,
    fieldpoints,
    vortexpathpoints,
    vortexcoreradii,
    vortexcirculations)

    # Compute the number of vortex path segments
    num_vpsegs = UInt32(1)  # UInt32(size(vortexpathpoints, 2)) - UInt32(1)

    # Compute the global index of the thread
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    # Initialize the return velocity array.
    velocity = SVector{3, Float32}(0, 0, 0)

    # Check if the thread index is out of bounds
    if idx <= size(fieldpoints, 2)
        # Get this thread's field point from the batch
        # If we needed more than 3 components, we would
        # use a for loop. 
        @inbounds fldpnt = SVector{3, Float32}(
            fieldpoints[1, idx],
            fieldpoints[2, idx],
            fieldpoints[3, idx])

        # Step through each vortex path segment.
        # We are using a while loop because the CUDA.jl
        # docs says this is more efficient than using a
        # for loop with a step interval.
        segindex = UInt32(1)
        while segindex <= num_vpsegs
            velocity = velocity .+ bs_uniform_trapezoidal_rule(
                Int32(100),  # Number of integrations steps
                fldpnt,
                vortexpathpoints,
                vortexcoreradii,
                vortexcirculations,
                segindex)

            segindex += UInt32(1)  # Advance the loop counter
        end

        @inbounds returnvelocities[:, idx] .= velocity
    end

    return nothing
end




################ Straight Vortex Test Case ################
#=
**Straight Vortex Test Case**
This is the start of a unit test of the
weighted_biot_savart_kernel function.

Let a infinitely long straight vortex be aligned with
the x-axis. We will evaluate the velocity at points
along the y-axis. 
=#
using Plots

# Set problem parameters
NUMVPSEGS = 200  # Number of vortex path segments
NUMFP = 1_000_000  # Number of field points
NUMTHREADS = 1024
NUMBLOCKS = ceil(Int, NUMFP / NUMTHREADS)

# Define the vortex path
vps = zeros(Float32, 3, NUMVPSEGS + 1)
vps[1, :] .= range(-1000, stop=1000, length=NUMVPSEGS + 1)
cuvps = CuArray{Float32}(vps)

# Define the vortex core
vcs = ones(Float32, NUMVPSEGS + 1)
cuvcs = CuArray{Float32}(vcs)

# Define the circulation at each path point
cirs = ones(Float32, NUMVPSEGS + 1)
cucirs = CuArray{Float32}(cirs)

# Define the field points
y = zeros(Float32, NUMFP)
y .= range(0, 20, length=NUMFP)  # end point is included in range
y[1] = 1e-3  # avoid divide by zero
fps = zeros(Float32, 3, NUMFP)
fps[2, :] = y
cufps = CuArray{Float32}(fps)

# Create the return velocities array
# CuArray is mutable!!!
curntvels = CuArray{Float32}(undef, 3, NUMFP)

# # Run the kernel function on the GPU
# # @device_code_warntype 
# # @device_code_llvm
# # @device_code_lowered
# @cuda blocks=NUMBLOCKS threads=NUMTHREADS weighted_biot_savart_kernel(
#     curntvels,
#     cufps,
#     cuvps,
#     cuvcs,
#     cucirs)

# Precompile the kernel
k = @cuda launch=false weighted_biot_savart_kernel(
    CuArray{Float32}(undef, 3, 1),
    CuArray{Float32}(undef, 3, 1),
    CuArray{Float32}(undef, 3, 1),
    CuArray{Float32}(undef, 1),
    CuArray{Float32}(undef, 1))

println("Max number of thread: ", CUDA.maxthreads(k))  # Queries the maximum amount of threads a kernel can use in a single block.
println("Register usage: ", CUDA.registers(k))  # Queries the register usage of a kernel.
println("Memory usage: ", CUDA.memory(k))  # Queries the local, shared and constant memory usage of a compiled kernel in bytes. Returns a named tuple.

# Run the kernel
k(curntvels, cufps, cuvps, cuvcs, cucirs;
    blocks=NUMBLOCKS, threads=NUMTHREADS)


################ Results ################

vel_num = Array(curntvels)
# vel_num = biot_savart(fp, vp, vc, cir)

vel_true = 1 ./ (2 .* pi .* y)
# println("vel_true = ", vel_true)
println("L2 Error (straight vortex): ", norm(vel_num[3, :] - vel_true))
println("L2 Error (straight vortex, removed first point): ", norm(vel_num[3, 2:end] - vel_true[2:end]))

plt = plot(y[1:1000:end], vel_num[3, 1:1000:end], markershape=:o, label="Numerical")
plot!(y[1:1000:end], vel_true[1:1000:end], markershape=:x, label="Analytical")
xlims!(0, 1)
display(plt)


################ Debug ################

# # Device function wrapper for debugging
# function dev_fn_wrapper(rtn,
#     fldpnt,
#     vortexpathpoints,
#     vortexcoreradii,
#     vortexcirculations)

#     fp = SVector{3, Float32}(fldpnt[1], fldpnt[2], fldpnt[3])
    
#     rtn .= bsintegrand(
#         fp,
#         vortexpathpoints,
#         vortexcoreradii,
#         vortexcirculations,
#         UInt32(1),  # segindex
#         Float32(0.5))  # ell

#     return nothing
# end

# cufp = cufps[:, 1]
# rtndebug = CUDA.zeros(Float32, 3)  # CuArray{Float32}(undef, 3)

# @show rtndebug

# # @device_code_warntype 
# # @device_code_llvm dump_module=true
# @device_code_warntype @cuda blocks=NUMBLOCKS threads=NUMTHREADS dev_fn_wrapper(
#     rtndebug,
#     cufp,
#     cuvps,
#     cuvcs,
#     cucirs)

# @show rtndebug

# nothing