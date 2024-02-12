using CUDA
using StaticArrays
using LinearAlgebra



################ Device Functions ################

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
# Vortex path model (For now) linear
function vpathmodel(vorpps, segidx, ell)
    @inbounds pnt1 = SVector{3, Float32}(
        vorpps[1, segidx],
        vorpps[2, segidx],
        vorpps[3, segidx])

    @inbounds pnt2 = SVector{3, Float32}(
        vorpps[1, segidx+UInt32(1)],
        vorpps[2, segidx+UInt32(1)],
        vorpps[3, segidx+UInt32(1)])

    seg = pnt2 .- pnt1
    
    return pnt1 .+ ell .* seg, seg ./ norm(seg)
end

function xi(fp, vorpps, segidx, ell)
    vppoint, vptan = vpathmodel(vorpps, segidx, ell)
    return fp .- vppoint, vptan
end

function vcoremodel(corerads, segidx, ell)
    @inbounds coreraddiff = corerads[segidx+UInt32(1)] - corerads[segidx]
    @inbounds return corerads[segidx] + ell * coreraddiff
end

function vcircmodel(circs, segidx, ell)
    @inbounds circdiff = circs[segidx+UInt32(1)] - circs[segidx]
    @inbounds return circs[segidx] + ell * circdiff
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
    # global i
    xiell, vptanell = xi(fp, vorpps, segidx, ell)
    xiellmag = norm(xiell)
    corell = vcoremodel(corerads, segidx, ell)
    circell = vcircmodel(circs, segidx, ell)
    direll = cross(vptanell, xiell)
    weightell = bsweightmodel(xiellmag / corell)
    return (weightell * circell / xiellmag^3) .* direll
end

# Define BS solver
function bs_uniform_trapezoidal_rule(numsteps, fp, vorpps, corerads, circs, segidx, ell)
    stepsize = Float32(1) / numsteps
    sol = MVector{3, Float32}(0, 0, 0)
    sol += bsintegrand(fp, vorpps, corerads, circs, segidx, Float32(0))
    sol += bsintegrand(fp, vorpps, corerads, circs, segidx, Float32(1))
    sol .*= Float32(0.5)

    # stepindex = UInt32(2)
    # while stepindex < numsteps
    #     sol .+= integrand(stepindex * stepsize, segindex)
    #     stepindex += UInt32(1)
    # end
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
    fieldpointsbatch,
    vortexpathpoints,
    vortexcoreradii,
    vortexcirculations)

    # Compute the number of vortex path segments
    numvpsegs = Int32(size(vortexpathpoints, 2) - 1)

    # Compute the global index of the thread
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    # Get this thread's field point from the batch
    @inbounds fieldpoint = SVector{3, Float32}(
        fieldpointsbatch[1, idx],
        fieldpointsbatch[2, idx],
        fieldpointsbatch[3, idx])

    # Initialize the velocity at the field point
    vel = MVector{3, Float32}(0, 0, 0)

    # Step through each vortex path segment.
    # We are using a while loop because the CUDA.jl
    # docs says this is more efficient than using a
    # for loop with a step interval.
    segindex = UInt32(1)
    while segindex <= numvpsegs

        vel .+= bs_uniform_trapezoidal_rule(
            UInt32(1000),  # Number of integrations steps
            fieldpoint,
            vortexpathpoints,
            vortexcoreradii,
            vortexcirculations,
            segindex,
            Float32(0.5))

        segindex += UInt32(1)  # Advance the loop counter
    end

    returnvelocities[:, idx] .= vel
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
numvpsegs = 2
vp = zeros(Float32, 3, numvpsegs + 1)
vp[1, :] .= range(-1000, stop=1000, length=numvpsegs + 1)
vp = CuArray{Float32}(vp)
vc = CUDA.ones(Float32, numvpsegs + 1)

# Define the circulation at each path point
cir = CUDA.ones(Float32, numvpsegs + 1)

# Define the field points
numfp = 10_000
y = zeros(Float32, numfp)
y .= range(0, 20, length=numfp)  # end point is included in range
y[1] = 1e-3  # avoid divide by zero
fp = zeros(Float32, 3, numfp)
fp[2, :] = y
fp = CuArray{Float32}(fp)

# Initialize velocity array
vel_num = zeros(3, numfp)
# println("size(vel_num) = ", size(vel_num))  # DEBUG

numblocks = 4
numthreads = 1024
fpbatchsize = numblocks * numthreads
numbatches = ceil(Int, size(fp, 2) / fpbatchsize)
fpbatchindex = 1
for fpbatchindex in 1:(numbatches-1)
    println("On batch ", fpbatchindex, " of ", numbatches, ".")

    # Create the return velocities array
    # CuArray is mutable!!!
    rtnv = CuArray{Float32}(undef, 3, fpbatchsize)
    # println("size(rtnv) = ", size(rtnv))  # DEBUG
    # display(rtnv)

    # Get a batch of: field points
    spanstart = 1 + ((fpbatchindex - 1) * fpbatchsize)
    # @show spanstart  # DEBUG
    spanend = fpbatchindex * fpbatchsize
    # @show spanend  # DEBUG
    fpbatch = fp[:, spanstart:spanend]

    println("\tSending to CUDA kernel...")
    # Call the kernel
    # @device_code_warntype 
    # @device_code_llvm
    # @device_code_lowered
    @cuda blocks=numblocks threads=numthreads weighted_biot_savart_kernel(
        rtnv,
        fpbatch,
        vp,
        vc,
        cir)

    # print("rtnv (after):")
    # display(rtnv)

    println("\tCopying to host...")
    vel_num[:, spanstart:spanend] .= Array(rtnv)
end

# Do last batch separately (whatever is left over)
spnstrt = 1 + ((numbatches - 1) * fpbatchsize)
fpbtch = fp[:, spnstrt:end]
rv = CuArray{Float32}(undef, 3, size(fpbtch, 2))
nblks = 2
nthrds = ceil(Int, size(fpbtch, 2) / nblks)
@cuda blocks=nblks threads=nthrds weighted_biot_savart_kernel(
    rv,
    fpbtch,
    vp,
    vc,
    cir)

vel_num[:, spnstrt:end] .= Array(rv)

vel_true = 1 ./ (2 .* pi .* y)
# println("vel_true = ", vel_true)
println("L2 Error (straight vortex): ", norm(vel_num[3, :] - vel_true))
println("L2 Error (straight vortex, removed first point): ", norm(vel_num[3, 2:end] - vel_true[2:end]))

plt = plot(y, vel_num[3, :], markershape=:o, label="Numerical")
plot!(y, vel_true, markershape=:x, label="Analytical")
# xlims!(0, .0625)
xlims!(0, 0.2)
display(plt)