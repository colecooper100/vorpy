using weighted_biot_savart_cuda

# ####################################################
# #===============================================
# This module implements the CUDA version of the
# weighted Biot-Savart solver. The solver loops
# over all the field points supplied to the
# method as a 3xN array and returns a vector of
# velocities equal in langth to the number of
# field points supplied.
# The method for computing the velocity at a
# single field point given a vortex path is
# implemented to run on the CPU or GPU. This,
# file handles the parallelization of the method
# (should any be available).
# ===============================================#
# using CUDA
# using StaticArrays
# using weighted_biot_savart_solver
# using utilities

# function u_wbs_1fp_cuda(rtnvel::AbstractArray{T, 2},
#                             fps::AbstractArray{T, 2},
#                             vpps::AbstractArray{T, 2},
#                             crads::AbstractArray{T, 1},
#                             circs::AbstractArray{T, 1},
#                             stepscalar::T) where {T<:AbstractFloat}
#     # Wrapper which turns u_wbs_1fp into a kernel function
    

#     # Compute the global index of the thread
#     idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

#     # Check if the thread index is in bounds
#     if idx <= size(fps, 2)
#         fp = getfp(fps, idx)
#         vel = u_wbs_1fp(fp,
#                     vpps,
#                     crads,
#                     circs,
#                     stepscalar)

#         @inbounds rtnvel[:, idx] .= vel
#     end

# return nothing
# end


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
# ####################################################

using LinearAlgebra

TYP = Float32
a = TYP[1, 2]
b = norm(a)
println("b: ", b)

# b = TYP[4, 5, 6]
# c = norm(a .- b)
# println("c: ", c)