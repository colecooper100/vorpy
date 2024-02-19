using CUDA
using StaticArrays
using LinearAlgebra
using BenchmarkTools

include("core_weight.jl")



################ Device functions ################

#=
I don't think CUDA supports nested functions,
so, the functions used in the kernel are
defined outside of it here.

Note: The code doesn't have to be physically
outside of the kernel, you just can't assume you
have access to any variables local to the outer
function.
=#

#============================================== 
---------------Vortex Path Model---------------
The vortex path is modeled as a piecewise linear
curve.

It is not clear to me how the integrator would
know the length of each vortex path segment.

I think the easier solution is to assume the
integrator will use \ell \in [0, 1] as the arc
length variable, and the vortex path model will
handle translating that to the real arc length. 

Input:
    - vpps: vortex path points
    - sidx: segment index
    - ell \in [0, 1]: arc length
Output:
    - vortex path point at ell
    - unit tangent vector at ell
==============================================#
function vpathmodel(vppnts, indx, ell)
    @inbounds pnt1 = SVector{3, Float32}(
        vppnts[1, indx],
        vppnts[2, indx],
        vppnts[3, indx])

    @inbounds seg = SVector{3, Float32}(
        vppnts[1, indx+1] - pnt1[1],
        vppnts[2, indx+1] - pnt1[2],
        vppnts[3, indx+1] - pnt1[3])

    return pnt1 .+ (ell .* seg), seg ./ norm(seg)
end

function xi(fp, vppnts, indx, ell)
    vpell, vtanell = vpathmodel(vppnts, indx, ell)
    return fp .- vpell, vtanell
end

# Core model simple linear interpolation
function vcoremodel(vcrads, indx, ell)
    @inbounds return vcrads[indx] + ell * (vcrads[indx+1] - vcrads[indx])
end

# Circulation model simple linear interpolation
function vcircmodel(vcircs, indx, ell)
    @inbounds return vcircs[indx] + ell * (vcircs[indx+1] - vcircs[indx])
end

# # No weight function
# bsweightmodel(x) = Float32(1)

# Bernstein polynomial weight function
bsweightmodel(x) = bernstein_polynomial_weight(x)

#============================================
---------------BS Integrator---------------
============================================#
# Define the integrand function
function bsintegrand(fp, vppnts, vcrads, vcircs, indx, ell)
    xiell, vtanell = xi(fp, vppnts, indx, ell)
    xiellmag = norm(xiell)
    corell = vcoremodel(vcrads, indx, ell)
    circell = vcircmodel(vcircs, indx, ell)
    direll = cross(vtanell, xiell)
    weightell = bsweightmodel((xiellmag / corell)^2)
    return (weightell * circell / xiellmag^3) .* direll
end

# Define BS solver
function bs_uniform_trapezoidal_rule(numsteps, fp, vppnts, vcrads, vcircs, indx)
    sol = bsintegrand(fp, vppnts, vcrads, vcircs, indx, Float32(0))
    sol = sol .+ bsintegrand(fp, vppnts, vcrads, vcircs, indx, Float32(1))
    sol = sol .* Float32(0.5)

    # Start stepindex at 2 because we already did
    # the first step and use 'less-than' becase we
    # already did the last step
    stepindex = UInt32(2)
    stepsize = Float32(1) / numsteps
    if stepsize < 1e-6
        @cuprintln("Warning: integrator stepsize is: ", stepsize)  # DEBUG
    end
    while stepindex < numsteps
        sol = sol .+ bsintegrand(fp, vppnts, vcrads, vcircs, indx, stepindex * stepsize)
        stepindex += UInt32(1)
    end

    # @cuprintln("typeof(sol): ", typeof(sol))  # DEBUG
    return sol .* (stepsize / Float32(4 * pi))
end


################ Biot-Savart kernel ################
#============================================
---------------velocity solver---------------
When we compute the velocity we will
devide the region into two parts: inside
some cutoff radius, we will integrate the BS
law, outside the cutoff we will use the
analytical solution for the velocity of an
infinitesimally thin vortex filament.
- We need to determine the cutoff radius
============================================#
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
    fps,
    vppnts,
    vcrads,
    vcircs)

    # Compute the number of vortex path segments
    num_vsegs = UInt32(size(vppnts, 2)) - UInt32(1)

    # Compute the global index of the thread
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    # # DEBUG
    # if idx == 1
    #     @cuprintln("Number of vortex path segments: ", num_vsegs)
    # end

    # Set the initial flow velocity
    velocity = SVector{3, Float32}(0, 0, 0)

    # Check if the thread index is in bounds
    if idx <= size(fps, 2)
        # Get this thread's field point from the batch
        # If we needed more than 3 components, we would
        # use a for loop for this.
        @inbounds fp = SVector{3, Float32}(
            fps[1, idx],
            fps[2, idx],
            fps[3, idx])

        # Step through each vortex path segment.
        # We are using a while loop because the CUDA.jl
        # docs says this is more efficient than using a
        # for loop with a step interval.
        indx = UInt32(1)
        while indx <= num_vsegs
            velocity = velocity .+ bs_uniform_trapezoidal_rule(
                UInt32(1_000),  # Number of integrations steps
                fp,
                vppnts,
                vcrads,
                vcircs,
                indx)
            
            # @cuprintln("typeof(velocity): ", typeof(velocity))  # DEBUG
            indx += UInt32(1)  # Advance the loop counter
        end

        @inbounds returnvelocities[:, idx] .= velocity
    end

    return nothing
end



################ User API ################

# Precompile the kernel
_biot_savart_solver = @cuda launch=false weighted_biot_savart_kernel(
    CuArray{Float32}(undef, 3, 1),
    CuArray{Float32}(undef, 3, 1),
    CuArray{Float32}(undef, 3, 1),
    CuArray{Float32}(undef, 1),
    CuArray{Float32}(undef, 1))

println("Max number of thread: ", CUDA.maxthreads(_biot_savart_solver))  # Queries the maximum amount of threads a kernel can use in a single block.
println("Register usage: ", CUDA.registers(_biot_savart_solver))  # Queries the register usage of a kernel.
println("Memory usage: ", CUDA.memory(_biot_savart_solver))  # Queries the local, shared and constant memory usage of a compiled kernel in bytes. Returns a named tuple.

function bs_solve(fieldpoints, vorpathpoints, vorcorrads, vorcircs)
    """
    bs_solve(fieldpoints, vorpathpoints, vorcorrads, vorcircs)

    Compute the flow velocity at the field points due to the
    vortex defined by vorpathpoints, vorcorrads, and vorcircs.

    # Arguments
    - fieldpoints: 3 x N array
    - vorpathpoints: 3 x M array
    - vorcorrads: M array
    - vorcircs: M array
    """
    num_threads = 1024
    num_fps = size(fieldpoints, 2)
    num_blocks = ceil(Int, num_fps / num_threads)

    ret_vels = CuArray{Float32}(undef, 3, num_fps)

    # Run the kernel function on the GPU
    _biot_savart_solver(
        ret_vels,
        CuArray{Float32}(fieldpoints),
        CuArray{Float32}(vorpathpoints),
        CuArray{Float32}(vorcorrads),
        CuArray{Float32}(vorcircs);
        blocks=num_blocks,
        threads=num_threads)

    return Array(ret_vels)
end



# ################ Straight Vortex Test Case ################
# #=
# **Straight Vortex Test Case**
# This is the start of a unit test of the
# weighted_biot_savart_kernel function.

# Let a infinitely long straight vortex be aligned with
# the x-axis. We will evaluate the velocity at points
# along the y-axis. 
# =#
# using Plots

# # Set problem parameters
# NUMVPSEGS = 1  # Number of vortex path segments
# NUMFP = 10_000_000  # Number of field points
# # NUMTHREADS = 1024
# # NUMBLOCKS = ceil(Int, NUMFP / NUMTHREADS)

# # Define the vortex path
# vps = zeros(Float32, 3, NUMVPSEGS + 1)
# vps[1, :] .= range(-1000, stop=1000, length=NUMVPSEGS + 1)
# cuvps = CuArray{Float32}(vps)

# # Define the vortex core
# vcrds = ones(Float32, NUMVPSEGS + 1) .* 2
# cuvcrds = CuArray{Float32}(vcrds)

# # Define the circulation at each path point
# cirs = ones(Float32, NUMVPSEGS + 1)
# cucirs = CuArray{Float32}(cirs)

# # Define the field points
# y = zeros(Float32, NUMFP)
# y .= range(0, 20, length=NUMFP)  # end point is included in range
# y[1] = 1e-3  # avoid divide by zero
# fps = zeros(Float32, 3, NUMFP)
# fps[2, :] = y
# cufps = CuArray{Float32}(fps)

# # Create the return velocities array
# # CuArray is mutable!!!
# curntvels = CuArray{Float32}(undef, 3, NUMFP)

# # # Run the kernel function on the GPU
# # # @device_code_warntype 
# # # @device_code_llvm
# # # @device_code_lowered
# # @device_code_warntype @cuda blocks=NUMBLOCKS threads=NUMTHREADS weighted_biot_savart_kernel(
# #     curntvels,
# #     cufps,
# #     cuvps,
# #     cuvcrds,
# #     cucirs)


# ################ Results ################

# # Test run the precompiled kernel
# vel_num = bs_solve(fps, vps, vcrds, cirs)

# # vel_num = Array(curntvels)
# # vel_num = biot_savart(fp, vp, vc, cir)

# vel_true = 1 ./ (2 .* pi .* y)
# # println("vel_true = ", vel_true)
# println("L2 Error (straight vortex): ", norm(vel_num[3, :] - vel_true))
# # println("L2 Error (straight vortex, removed first point): ", norm(vel_num[3, 2:end] - vel_true[2:end]))
# println("Root mean squared error (straight vortex): ", sqrt(mean((vel_num[3, :] - vel_true).^2)))
# # println("Root mean squared error (straight vortex, removed first point): ", sqrt(mean((vel_num[3, 2:end] - vel_true[2:end]).^2)))

# stride = 1000
# pltvelmag = plot(y[1:stride:end], vel_num[3, 1:stride:end], markershape=:o, label="Numerical")
# # plot!(pltvelmag, y[1:stride:end], vel_true[1:stride:end], markershape=:x, label="Analytical")
# xlabel!("y")
# ylabel!("Velocity")
# # xlims!(0, 1)
# title!(pltvelmag, "Straight Vortex Test Case,\nVelocity at every $(stride)th Field Point")
# display(pltvelmag)

# # plterror = plot(y[1:stride:end], abs.(vel_num[3, 1:stride:end] .- vel_true[1:stride:end]), markershape=:x, label="Error")
# # title!(plterror, "Straight Vortex Test Case - Absolute Error")
# # xlabel!("y")
# # ylabel!("Absolute Error")
# # ylims!(0, .1)
# # display(plterror)


################ Debug ################

# # Device function wrapper for debugging
# function dev_fn_wrapper(rtn,
#     fldpnt,
#     vortexpathpoints,
#     vortexcoreradii,
#     vortexcirculations)

#     # fp = SVector{3, Float32}(fldpnt[1], fldpnt[2], fldpnt[3])
    
#     rtn[1] = bernstein_polynomial_weight(Float32(0.5))

#     return nothing
# end

# cufp = cufps[:, 1]
# rtndebug = CUDA.zeros(Float32, 1)  # CuArray{Float32}(undef, 3)

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