using CUDA: @cuda, blockIdx, blockDim, CuArray, threadIdx, CUDA
using StaticArrays: SVector

include("biot_savart_solver_vortex_line.jl")


function weighted_biot_savart_kernel_cuda!(rtnvelocities, fps, vpps, vcrs, cirs)

    # Compute the global index of the thread
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    # Check if the thread index is in bounds
    if idx <= size(fps, 2)
        # Get this thread's field point from the batch
        # If we needed more than 3 components, we would
        # use a for loop for this.
        @inbounds fp = SVector{3, Float32}(
            fps[1, idx],
            fps[2, idx],
            fps[3, idx])
        
        velocity = velocity_field_point(fp, vpps, vcrs, cirs)

        @inbounds rtnvelocities[:, idx] .= velocity
    end

    return nothing
end



################ User API ################

# Precompile the kernel
_precompiled_weight_bs_kernel = @cuda launch=false weighted_biot_savart_kernel_cuda!(
    CuArray{Float32}(undef, 3, 1),
    CuArray{Float32}(undef, 3, 1),
    CuArray{Float32}(undef, 3, 1),
    CuArray{Float32}(undef, 1),
    CuArray{Float32}(undef, 1))

println("Max number of thread: ", CUDA.maxthreads(_precompiled_weight_bs_kernel))  # Queries the maximum amount of threads a kernel can use in a single block.
println("Register usage: ", CUDA.registers(_precompiled_weight_bs_kernel))  # Queries the register usage of a kernel.
println("Memory usage: ", CUDA.memory(_precompiled_weight_bs_kernel))  # Queries the local, shared and constant memory usage of a compiled kernel in bytes. Returns a named tuple.

function bs_solve_cuda(fieldpoints, vorpathpoints, vorcorrads, vorcircs)
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
    num_fps = size(fieldpoints, 2)


    # Create an array to store the return velocities
    ret_vels = CuArray{Float32}(undef, 3, num_fps)

    # Set the number of threads and blocks
    num_threads = CUDA.maxthreads(_precompiled_weight_bs_kernel)
    num_blocks = ceil(Int, num_fps / num_threads)
    # Run the kernel function on the GPU
    _precompiled_weight_bs_kernel(
        ret_vels,
        CuArray{Float32}(fieldpoints),
        CuArray{Float32}(vorpathpoints),
        CuArray{Float32}(vorcorrads),
        CuArray{Float32}(vorcircs);
        blocks=num_blocks,
        threads=num_threads)

    return Array(ret_vels)
end



# ################### DEBUGGING ###################
# using Plots
# using BenchmarkTools

# #==============================================
# -----------Straight Vortex Test Case-----------
# We define a Lamb-Oseen vortex aligned with the
# z-axis.

# If our coordinate system has +z running from
# left-to-right, then +x is into the page and +y
# runs from down-to-up. We compute the flow
# velocity at points along the +x-axis at the
# origin, i.e., $\vec r = (x, 0, 0)$.
# ==============================================#
# # Set problem parameters
# VDOMAIN = [-1000, 1000]  # Vortex path domain
# FPDOMAIN = [0, 20]  # Field points domain
# NUMVPSEGS = 3  # Number of vortex path segments
# NUMFP = 10_000  # Number of field points
# println("VDOMAIN = ", VDOMAIN)
# println("FPDOMAIN = ", FPDOMAIN)
# println("NUMVPSEGS = ", NUMVPSEGS)
# println("NUMFP = ", NUMFP)

# # Define the vortex
# vpps = zeros(Float32, 3, NUMVPSEGS + 1)  # Path points
# vpps[3, :] .= range(VDOMAIN[1], stop=VDOMAIN[2], length=NUMVPSEGS + 1)
# # Core diameters
# vcrs = ones(Float32, NUMVPSEGS + 1) # Float32.(collect(axes(vpps, 2)))
# # Circulations
# cirs = ones(Float32, NUMVPSEGS + 1)  # Float32.(collect(axes(vpps, 2)))

# # Define the field points
# x = zeros(Float32, NUMFP)
# x .= range(FPDOMAIN[1], FPDOMAIN[2], length=NUMFP)  # end point is included in range
# if x[1] == 0 
#     x[1] = 1e-3  # avoid divide by zero
# end
# fps = zeros(Float32, 3, NUMFP)
# fps[1, :] .= x



# ####### Run #######
# # # Benchmark 10_000 field points
# # # CUDA: 7.461 ms, 118.67 KB, 41 allocs
# # # CPU: 130.879 ms, 237.08 KB, 33 allocs
# # # @benchmark CUDA.@sync bs_solve($fps, $vpps, $vcrs, $cirs; device="cuda")
# # @benchmark Threads.@sync bs_solve($fps, $vpps, $vcrs, $cirs; device="cpu")

# vel_num = bs_solve_cuda(fps, vpps, vcrs, cirs)
# plot(x[1:100_000:end], vel_num[2, 1:100_000:end], markershape=:x, label="Numerical")

# # Analytical solution (infinitesimally thin vortex)
# vel_true_nocore = Float32.(1 ./ (2 .* pi .* x))

# # Analytical solution (finite core)
# vel_true_core = Float32.(1 ./ (2 .* pi .* x) .* (1 .- exp.(-x.^2 ./ (2 * vcrs[1]^2))))

# # println("vel_true = ", vel_true)
# println("Total L2 error (straight vortex): ", norm(vel_num[2, :] - vel_true_core))
# println("Avarage L2 error (straight vortex): ", sqrt(mean((vel_num[2, :] - vel_true_core).^2)))

# # println("vel_true_core = ")  # DEBUG
# # display(vel_true_core)  # DEBUG
# # println("vel_num = ")  # DEBUG
# # display(vel_num[3, :])  # DEBUG

# stride = 100
# pltvelmag = plot(x[1:stride:end], vel_num[2, 1:stride:end], markershape=:o, label="Numerical")
# plot!(pltvelmag, x[1:stride:end], vel_true_nocore[1:stride:end], markershape=:x, label="Analytical (No Core)")
# plot!(pltvelmag, x[1:stride:end], vel_true_core[1:stride:end], markershape=:x, label="Analytical (Core)")
# xlabel!("y")
# ylabel!("Velocity")
# # xlims!(0, 1)
# ylims!(0, .1)
# title!(pltvelmag, "Straight Vortex Test Case,\nVelocity at every $(stride)th Field Point")
# display(pltvelmag)

# plterror = plot(x[1:stride:end], abs.(vel_num[2, 1:stride:end] .- vel_true_core[1:stride:end]), markershape=:x, label="Error")
# title!(plterror, "Straight Vortex Test Case - Absolute Error")
# xlabel!("y")
# ylabel!("Absolute Error")
# # ylims!(0, .1)

# plot(pltvelmag, plterror, layout=(2, 1), legend=false)