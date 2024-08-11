#===============================================
This module implements the CUDA version of the
weighted Biot-Savart solver. The solver loops
over all the field points supplied to the
method as a 3xN array and returns a vector of
velocities equal in langth to the number of
field points supplied.
The method for computing the velocity at a
single field point given a vortex path is
implemented to run on the CPU or GPU. This,
file handles the parallelization of the method
(should any be available).
===============================================#

module weighted_biot_savart_cuda

using CUDA
using StaticArrays
using weighted_biot_savart_solver
using utilities

export u_wbs_cuda, precompile_u_wbs_cuda

function u_wbs_1fp_cuda(rtnvel::AbstractArray{T, 2},
                            fps::AbstractArray{T, 2},
                            vpps::AbstractArray{T, 2},
                            crads::AbstractArray{T, 1},
                            circs::AbstractArray{T, 1},
                            stepscalar::T) where {T<:AbstractFloat}
    # Wrapper which turns u_wbs_1fp into a kernel function
    

    # Compute the global index of the thread
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    # Check if the thread index is in bounds
    if idx <= size(fps, 2)
        fp = getfp(fps, idx)
        vel = u_wbs_1fp(fp,
                    vpps,
                    crads,
                    circs,
                    stepscalar)

        @inbounds rtnvel[:, idx] .= vel
    end

return nothing
end


# Define the GPU kernel function
function u_wbs_cuda(fps::AbstractArray{T1, 2},
    vpps::AbstractArray{T1, 2},
    crads::AbstractArray{T1, 1},
    circs::AbstractArray{T1, 1},
    stepscalar::T1=T1(0.25);
    numthreads::T2=UInt16(350)) where {T1<:AbstractFloat, T2<:Integer}
    #============================================
    `numthreads` sets the number of threads to
    use. The more threads used the faster the
    computation. However, the max number of
    threads that can be used will depends on
    several factors such as the physical limits
    of the GPU (threads per block, registers per
    block, etc.) and the size of the problem.
    The default value is 300.
    ============================================#

    numfps = size(fps, 2)
    curtnvel = CUDA.zeros(T1, 3, numfps)

    # println("* u_wbs_cuda_test *")
    # println("Field points: ", fps)  # DEBUG
    # println("Vortex path points: ", vpps)  # DEBUG
    # println("Core radii: ", crads)  # DEBUG
    # println("Circulations: ", circs)  # DEBUG
    # println("Step scalar: ", stepscalar)  # DEBUG

    # Run the kernel function on the GPU
    # Make sure to set the number of threads and blocks
    # The number of blocks tells the GPU how much work
    # it has to do. If you have one block, and one thread,
    # then the WBS function is called for the first field.
    # If you had 10 field points, for the same example,
    # you would need 10 blocks (or 1 block of 10 threads).
    
    # numthreads = 1
    numblocks = ceil(T2, numfps / numthreads)
    # @device_code_warntype  # use this for debugging
    @cuda threads=numthreads blocks=numblocks u_wbs_1fp_cuda(curtnvel,
                                                                CuArray{T1}(fps),
                                                                CuArray{T1}(vpps),
                                                                CuArray{T1}(crads),
                                                                CuArray{T1}(circs),
                                                                T1(stepscalar))

    return Array(curtnvel)
end


function precompile_u_wbs_cuda(TYP::DataType=Float32)
    # TYP should be Float32 usually (because GPUs
    # process single precision floats faster)

    precompiled_u_wbs_1fp_cuda = @cuda launch=false u_wbs_1fp_cuda(CuArray{TYP}(undef, 3, 1),
                                                            CuArray{TYP}(undef, 3, 1),
                                                            CuArray{TYP}(undef, 3, 1),
                                                            CuArray{TYP}(undef, 1),
                                                            CuArray{TYP}(undef, 1),
                                                            TYP(0.0))

    println("Max number of threads: ", CUDA.maxthreads(precompiled_u_wbs_1fp_cuda))  # Queries the maximum amount of threads a kernel can use in a single block.
    println("Register usage: ", CUDA.registers(precompiled_u_wbs_1fp_cuda))  # Queries the register usage of a kernel.
    println("Memory usage: ", CUDA.memory(precompiled_u_wbs_1fp_cuda))

    function precompiled_u_wbs(fieldpoints::AbstractArray{T, 2},
                            vorpathpoints::AbstractArray{T, 2},
                            cordradii::AbstractArray{T, 1},
                            circulations::AbstractArray{T, 1};
                            stepsizescalar::T=T(0.25)) where {T<:AbstractFloat}
        
        num_fps = size(fieldpoints, 2)

        # Create an array to store the return velocities
        ret_vels = CuArray{T}(undef, 3, num_fps)

        # Set the number of threads and blocks
        num_threads = CUDA.maxthreads(precompiled_u_wbs_1fp_cuda)
        num_blocks = ceil(Int, num_fps / num_threads)
        # Run the kernel function on the GPU
        precompiled_u_wbs_1fp_cuda(ret_vels,  
                            CuArray{T}(fieldpoints),
                            CuArray{T}(vorpathpoints),
                            CuArray{T}(cordradii),
                            CuArray{T}(circulations),
                            T(stepsizescalar);
                            blocks=num_blocks,
                            threads=num_threads)

        return Array(ret_vels)
    end

    println("* u_wbs_cuda precompiled!")

    return precompiled_u_wbs
end

# ###### Precompile the GPU kernel function ######
# precompiled_u_wbs_1fp_cuda = @cuda launch=false u_wbs_1fp_cuda(CuArray{Float32}(undef, 3, 1),
#                                                                 CuArray{Float32}(undef, 3, 1),
#                                                                 CuArray{Float32}(undef, 3, 1),
#                                                                 CuArray{Float32}(undef, 1),
#                                                                 CuArray{Float32}(undef, 1),
#                                                                 Float32(1))

# println("Max number of threads: ", CUDA.maxthreads(precompiled_u_wbs_1fp_cuda))  # Queries the maximum amount of threads a kernel can use in a single block.
# println("Register usage: ", CUDA.registers(precompiled_u_wbs_1fp_cuda))  # Queries the register usage of a kernel.
# println("Memory usage: ", CUDA.memory(precompiled_u_wbs_1fp_cuda))  # Queries the local, shared and constant memory usage of a compiled kernel in bytes. Returns a named tuple.


# ###### User API ######
# function u_wbs_cuda(fieldpoints::AbstractArray{T, 2},
#                         vorpathpoints::AbstractArray{T, 2},
#                         cordradii::AbstractArray{T, 1},
#                         circulations::AbstractArray{T, 1};
#                         stepsizescalar::T=T(0.25)) where {T<:AbstractFloat}
    
#     num_fps = size(fieldpoints, 2)

#     # Create an array to store the return velocities
#     ret_vels = CuArray{T}(undef, 3, num_fps)

#     # Set the number of threads and blocks
#     num_threads = CUDA.maxthreads(precompiled_u_wbs_1fp_cuda)
#     num_blocks = ceil(Int, num_fps / num_threads)
#     # Run the kernel function on the GPU
#     precompiled_u_wbs_1fp_cuda(
#     ret_vels,
#     CuArray{T}(fieldpoints),
#     CuArray{T}(vorpathpoints),
#     CuArray{T}(cordradii),
#     CuArray{T}(circulations),
#     T(stepsizescalar);
#     blocks=num_blocks,
#     threads=num_threads)

#     return Array(ret_vels)
# end

end # module weighted_biot_savart_cuda
