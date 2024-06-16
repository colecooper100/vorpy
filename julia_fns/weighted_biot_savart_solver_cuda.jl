#===============================================
This script implements a GPU version of the
weighted Biot-Savart solver. The solver loops
over all field points and returns a velocity
vector of langth equal to the number of field
points.
The reason this file contains the code for
looping through field points is to allow for
parallelization. The way the CPU has to be
parallelized is different from the GPU, so we
have separate implementations for each. 
===============================================#


###### Import modules and local scrips ######
using CUDA: @cuda, blockIdx, blockDim, CuArray, threadIdx, CUDA
using StaticArrays: SVector, SMatrix

# Import the function that calculates the velocity
include(string(WEIGHTED_BIOT_SAVART_SOLVER_ONE_FIELD_POINT, "/weighted_biot_savart_solver_one_field_point.jl"))


###### GPU kernel function ######
function _weighted_biot_savart_solver_kernel(rtnvelocities::AbstractArray{T, 2},
                                                fieldpoints::AbstractArray{T, 2},
                                                vorpathpoints::AbstractArray{T, 2},
                                                cordradii::AbstractArray{T, 1},
                                                circulations::AbstractArray{T, 1},
                                                stepsizescalar::T) where {T<:AbstractFloat}

    # Compute the global index of the thread
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    # Check if the thread index is in bounds
    if idx <= size(fieldpoints, 2)
        # Get this thread's field point from the batch
        # If we needed more than 3 components, we would
        # use a for loop for this.
        @inbounds fp = SVector{3, Float32}(
            fieldpoints[1, idx],
            fieldpoints[2, idx],
            fieldpoints[3, idx])
        
        velocity = weighted_biot_savart_for_one_field_point(fp,
                                                            vorpathpoints,
                                                            cordradii,
                                                            circulations,
                                                            stepsizescalar)

        @inbounds rtnvelocities[:, idx] .= velocity
    end

    return nothing
end


###### Precompile the GPU kernel function ######
_precompiled_wbs_kernel = @cuda launch=false _weighted_biot_savart_solver_kernel(
    CuArray{Float32}(undef, 3, 1),
    CuArray{Float32}(undef, 3, 1),
    CuArray{Float32}(undef, 3, 1),
    CuArray{Float32}(undef, 1),
    CuArray{Float32}(undef, 1),
    Float32(1))

println("Max number of thread: ", CUDA.maxthreads(_precompiled_wbs_kernel))  # Queries the maximum amount of threads a kernel can use in a single block.
println("Register usage: ", CUDA.registers(_precompiled_wbs_kernel))  # Queries the register usage of a kernel.
println("Memory usage: ", CUDA.memory(_precompiled_wbs_kernel))  # Queries the local, shared and constant memory usage of a compiled kernel in bytes. Returns a named tuple.


###### User API ######
function weighted_biot_savart_solver_cuda(fieldpoints::AbstractArray{T, 2},
                                            vorpathpoints::AbstractArray{T, 2},
                                            cordradii::AbstractArray{T, 1},
                                            circulations::AbstractArray{T, 1};
                                            stepsizescalar::T=0.5) where {T<:AbstractFloat}
    num_fps = size(fieldpoints, 2)

    # Create an array to store the return velocities
    ret_vels = CuArray{T}(undef, 3, num_fps)

    # Set the number of threads and blocks
    num_threads = CUDA.maxthreads(_precompiled_wbs_kernel)
    num_blocks = ceil(Int, num_fps / num_threads)
    # Run the kernel function on the GPU
    _precompiled_wbs_kernel(
        ret_vels,
        CuArray{T}(fieldpoints),
        CuArray{T}(vorpathpoints),
        CuArray{T}(cordradii),
        CuArray{T}(circulations),
        T(stepsizescalar);
        blocks=num_blocks,
        threads=num_threads)

    return Array(ret_vels)
end
