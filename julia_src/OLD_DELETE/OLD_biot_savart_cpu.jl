using StaticArrays: SVector
using LinearAlgebra: norm, cross
# using BenchmarkTools

include("biot_savart_integrator.jl")


function weighted_biot_savart_kernel_cpu!(
    returnvelocities,
    fps,
    vppnts,
    vcrads,
    vcircs)

    # Compute the number of vortex path segments
    num_vsegs = UInt32(size(vppnts, 2)) - UInt32(1)
    # println("num_vsegs = ", num_vsegs)  # DEBUG

    # println("size(fps) = ", size(fps))  # DEBUG

    for idx in axes(fps, 2)
        # println("idx = ", idx)  # DEBUG

        # Set the initial flow velocity
        velocity = SVector{3, Float32}(0, 0, 0)

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
        sindx = UInt32(1)
        while sindx <= num_vsegs
            velocity = velocity .+ bs_uniform_trapezoidal_rule(
                UInt32(10),  # Number of integrations steps
                fp,
                vppnts,
                vcrads,
                vcircs,
                sindx)
            
            # @cuprintln("typeof(velocity): ", typeof(velocity))  # DEBUG
            sindx += UInt32(1)  # Advance the loop counter

            # println("velocity = ", velocity)  # DEBUG
            @inbounds returnvelocities[:, idx] .= velocity
        end
    end

    return nothing
end


################ User API ################

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

    ret_vels = zeros(Float32, 3, size(fieldpoints, 2)...)
    # println("ret_vels = ", ret_vels)  # DEBUG

    # Run the kernel function on the GPU
    weighted_biot_savart_kernel_cpu!(
        ret_vels,
        Array{Float32}(fieldpoints),
        Array{Float32}(vorpathpoints),
        Array{Float32}(vorcorrads),
        Array{Float32}(vorcircs))

    return ret_vels
end

