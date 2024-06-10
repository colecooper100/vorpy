###### Import modules and local scrips ######
using StaticArrays: SVector
using LinearAlgebra: norm, cross

# environment_variables.jl sets the path varibles
# used to call other scripts
# pwd() returns the "present working directory". For
# this project, pwd() should return the path to the
# vorpy directory
include(string(pwd(), "/julia_src/environment_variables.jl"))
include(string(WEIGHTED_BIOT_SAVART_INTEGRATOR, "/weighted_biot_savart_integrator.jl"))


###### Function for extracting a path segment ######
# This is mainly needed for the GPU
function _getseg(vpps::AbstractArray{T, 2},
                    crads::AbstractArray{T, 1},
                    circs::AbstractArray{T, 1},
                    indx::Integer)::Tuple{SVector{3, T}, SVector{3, T}, T, T, T, T} where {T<:AbstractFloat}
    # Get starting point of segment
    @inbounds vpp1 = SVector{3, T}(
        vpps[1, indx],
        vpps[2, indx],
        vpps[3, indx])

    # Get the ending point of segment
    @inbounds vpp2 = SVector{3, T}(
        vpps[1, indx+1],
        vpps[2, indx+1],
        vpps[3, indx+1])

    @inbounds return vpp1, vpp2, crads[indx], crads[indx+1], circs[indx], circs[indx+1]
end


###### Function for computing flow velocity at ONE field point ######
#===================================================
We compute the flow velocity at a field point using
the Biot-Savart law. The vortex path is defined by a
series of path points. Additionally, at each point
we define the core diameter, and circulation.

This function loops through each segment of the
vortex path summing the velocities of each path
segment.

The function computes a stepsize for the integrator,
then passes the step size, field point, and segment
to the integrator, and the integrator returns the
velocity for that segment.
===================================================#
function weighted_biot_savart_for_one_field_point(fp::SVector{3, T},
                                                    vpps::AbstractArray{T, 2},
                                                    crads::AbstractArray{T, 1},
                                                    circs::AbstractArray{T, 1}) where {T<:AbstractFloat}
    # Compute the number of vortex path segments
    num_vsegs = UInt32(size(vpps, 2)) - UInt32(1)

    # Initialize the return velocity
    rtn_vel = SVector{3, Float32}(0, 0, 0)

    # Step through each vortex path segment.
    # We are using a while loop because the CUDA.jl
    # docs says this is more efficient than using a
    # for loop with a step interval.
    # I WOULD LIKE TO KNOW IF THIS IS TRUE!
    segindx = UInt32(1)
    while segindx <= num_vsegs
        # Get segment
        seg = _getseg(vpps, crads, circs, segindx)

        # Determine an appropriate step size for the
        # integrator.
        #==========================================
        The stepsize of the integrator is a
        fraction of the average core diameter of
        the segment.
        THE stepsize HAS A SIGNIFICANT IMPACT ON
        PERFORMANCE (OBVIOUSLY). I HAVE FOUND
        THAT A stepsize OF 1 CORE DIAMETER IS THE
        LARGEST THAT CAN BE USED WITHOUT NEGATIVELY
        AFFECTING THE ACCURACY OF THE SOLUTION.
        ==========================================#
        # Rough estimate of core diameter for determining
        # step size
        avg_crad = sum(seg[3:4]) / Float32(2)
        STEP_SIZE_SCALAR = Float32(0.5)
        stepsize = STEP_SIZE_SCALAR * avg_crad

        # Compute the velocity for the segment
        segvel = wbs_integrator(stepsize, fp, seg...)
        
        # Add the segment velocity to the accumulated
        # velocity
        rtn_vel = rtn_vel .+ segvel
        segindx += UInt32(1)  # Advance the loop counter
    end

    return rtn_vel
end