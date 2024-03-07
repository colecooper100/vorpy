using CUDA: @cuda, blockIdx, blockDim, CuArray, threadIdx, CUDA
using StaticArrays: SVector
using LinearAlgebra: norm, cross
# using BenchmarkTools

include("biot_savart_segment_integrator.jl")
bs_integrator(stepsize, fp, vpp1, vpp2, vcr1, vcr2, cir1, cir2) = bs_nonuniform_trapezoidal_rule_segment(stepsize, fp, vpp1, vpp2, vcr1, vcr2, cir1, cir2)


# Get a path segment
function get_segment(vpps, vcrs, cirs, indx)
    # Get starting point of segment
    @inbounds vpp1 = SVector{3, Float32}(
        vpps[1, indx],
        vpps[2, indx],
        vpps[3, indx])

    # Get the ending point of segment
    @inbounds vpp2 = SVector{3, Float32}(
        vpps[1, indx+1],
        vpps[2, indx+1],
        vpps[3, indx+1])

    @inbounds return vpp1, vpp2, vcrs[indx], vcrs[indx+1], cirs[indx], cirs[indx+1]
end

function weighted_biot_savart_kernel_cuda!(rtnvelocities, fps, vpps, vcrs, cirs)

    # Compute the number of vortex path segments
    num_vsegs = UInt32(size(vpps, 2)) - UInt32(1)

    # Compute the global index of the thread
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

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
        segindx = UInt32(1)

        # THE STEPSIZE HAS A SIGNIFICANT IMPACT ON
        # PERFORMANCE (OBVIOUSLY). I HAVE FOUND
        # THAT A STEPSIZE OF 1 IS THE LARGEST THAT
        # CAN BE USED WITHOUT OVERLY AFFECTING THE
        # ACCURACY OF THE SOLUTION.
        STEPSIZE = Float32(1)  # DEBUG
        while segindx <= num_vsegs
            segvel = bs_integrator(STEPSIZE, fp, get_segment(vpps, vcrs, cirs, segindx)...)
            velocity = velocity .+ segvel
            segindx += UInt32(1)  # Advance the loop counter
        end

        @inbounds rtnvelocities[:, idx] .= velocity
    end

    return nothing
end



################ User API ################

# Precompile the kernel
_biot_savart_solver = @cuda launch=false weighted_biot_savart_kernel_cuda!(
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
    num_threads = CUDA.maxthreads(_biot_savart_solver)
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



################### DEBUGGING ###################
using Plots
using BenchmarkTools

#==============================================
-----------Straight Vortex Test Case-----------
We define a Lamb-Oseen vortex aligned with the
z-axis.

If our coordinate system has +z running from
left-to-right, then +x is into the page and +y
runs from down-to-up. We compute the flow
velocity at points along the +x-axis at the
origin, i.e., $\vec r = (x, 0, 0)$.
==============================================#
# Set problem parameters
VDOMAIN = [-1000, 1000]  # Vortex path domain
FPDOMAIN = [0, 20]  # Field points domain
NUMVPSEGS = 3  # Number of vortex path segments
NUMFP = 10_000  # Number of field points
println("VDOMAIN = ", VDOMAIN)
println("FPDOMAIN = ", FPDOMAIN)
println("NUMVPSEGS = ", NUMVPSEGS)
println("NUMFP = ", NUMFP)

# Define the vortex
vpps = zeros(Float32, 3, NUMVPSEGS + 1)  # Path points
vpps[3, :] .= range(VDOMAIN[1], stop=VDOMAIN[2], length=NUMVPSEGS + 1)
# Core diameters
vcrs = ones(Float32, NUMVPSEGS + 1) # Float32.(collect(axes(vpps, 2)))
# Circulations
cirs = ones(Float32, NUMVPSEGS + 1)  # Float32.(collect(axes(vpps, 2)))

# Define the field points
x = zeros(Float32, NUMFP)
x .= range(FPDOMAIN[1], FPDOMAIN[2], length=NUMFP)  # end point is included in range
if x[1] == 0 
    x[1] = 1e-3  # avoid divide by zero
end
fps = zeros(Float32, 3, NUMFP)
fps[1, :] .= x



###### Run on the CPU ######




# ###### Run on the GPU ######

# @benchmark CUDA.@sync bs_solve($fps, $vpps, $vcrs, $cirs)
# println("----------------sol----------------")
# println(sol)
vel_num = bs_solve(fps, vpps, vcrs, cirs)
plot(x[1:100_000:end], vel_num[2, 1:100_000:end], markershape=:x, label="Numerical")

# Analytical solution (infinitesimally thin vortex)
vel_true_nocore = Float32.(1 ./ (2 .* pi .* x))

# Analytical solution (finite core)
vel_true_core = Float32.(1 ./ (2 .* pi .* x) .* (1 .- exp.(-x.^2 ./ (2 * vcrs[1]^2))))

# println("vel_true = ", vel_true)
# println("Total L2 error (straight vortex): ", norm(vel_num[3, :] - vel_true_core))
# println("Avarage L2 error (straight vortex): ", sqrt(mean((vel_num[3, :] - vel_true_core).^2)))

# println("vel_true_core = ")  # DEBUG
# display(vel_true_core)  # DEBUG
# println("vel_num = ")  # DEBUG
# display(vel_num[3, :])  # DEBUG

stride = 100
pltvelmag = plot(x[1:stride:end], vel_num[2, 1:stride:end], markershape=:o, label="Numerical")
plot!(pltvelmag, x[1:stride:end], vel_true_nocore[1:stride:end], markershape=:x, label="Analytical (No Core)")
plot!(pltvelmag, x[1:stride:end], vel_true_core[1:stride:end], markershape=:x, label="Analytical (Core)")
xlabel!("y")
ylabel!("Velocity")
# xlims!(0, 1)
ylims!(0, .1)
title!(pltvelmag, "Straight Vortex Test Case,\nVelocity at every $(stride)th Field Point")
display(pltvelmag)

# plterror = plot(y[1:stride:end], abs.(vel_num[3, 1:stride:end] .- vel_true[1:stride:end]), markershape=:x, label="Error")
# title!(plterror, "Straight Vortex Test Case - Absolute Error")
# xlabel!("y")
# ylabel!("Absolute Error")
# ylims!(0, .1)

# # display(plterror)