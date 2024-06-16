###### Import modules and local scrips ######
using StaticArrays: SVector
using LinearAlgebra: norm, cross

# Include the utility functions for debugging
# Comment out all timing and debugging code
# before running on the GPU!
include(string(UTILITY_FUNCTIONS, "/utility_functions.jl"))  # DEBUG
# t0 = time_ns()  # TIMING
# elapsed_time(t0)  # TIMING

tscriptstart = time_ns()  # TIMING
# Include the integrator
include(string(WEIGHTED_BIOT_SAVART_INTEGRATOR, "/weighted_biot_savart_integrator.jl"))
# println("-> Elapsed time for importing weighted_biot_savart_integrator.jl: ", elapsed_time(tscriptstart))  # DEBUG


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
                                                    circs::AbstractArray{T, 1},
                                                    stepsizescalar::T) where {T<:AbstractFloat}
    # println("Inside weighted_biot_savart_for_one_field_point")  # DEBUG
    # println("vpps:", vpps)  # DEBUG

    # Compute the number of vortex path segments
    num_vsegs = UInt32(size(vpps, 2)) - UInt32(1)
    
    # println("num_vsegs: ", num_vsegs)  # DEBUG

    # Initialize the return velocity
    rtn_vel = SVector{3, Float32}(0, 0, 0)

    # twhilestart = time_ns()  # TIMING
    # loopcounter = UInt32(0)  # DEBUG

    # Step through each vortex path segment.
    # We are using a while loop because the CUDA.jl
    # docs says this is more efficient than using a
    # for loop with a step interval.
    # I WOULD LIKE TO KNOW IF THIS IS TRUE!
    segindx = UInt32(1)
    while segindx <= num_vsegs

        # if loopcounter <= 3
        #     tloopstart = time_ns()  # TIMING
        #     loopcounter += UInt32(1)
        # end

        # Get segment
        seg = _getseg(vpps, crads, circs, segindx)

        # println("seg: ", seg)  # DEBUG

        # Determine an appropriate step size for the
        # integrator.
        #==========================================
        The stepsize of the integrator is a
        fraction of the average core radius of
        the segment.
        stepsize HAS A SIGNIFICANT IMPACT ON
        PERFORMANCE AND ACCURACY (OBVIOUSLY).
        ==========================================#
        # # Rough estimate of core diameter for determining
        # # step size
        # avg_crad = sum(seg[3:4]) / Float32(2)
        # STEP_SIZE_SCALAR = Float32(0.5)
        # stepsize = STEP_SIZE_SCALAR * avg_crad
        #
        # Get the minimum core diameter
        mincrad = min(seg[3], seg[4])
        stepsize = stepsizescalar * mincrad

        # Compute the velocity for the segment
        segvel = wbs_integrator(stepsize, fp, seg...)
        # Add the segment velocity to the accumulated
        # velocity
        rtn_vel = rtn_vel .+ segvel

        segindx += UInt32(1)  # Advance the loop counter

        # if loopcounter <= 3
        #     println("stepsize: ", stepsize)  # DEBUG
        #     # println("--> Time computing velocity for segment ", segindx, ": ", elapsed_time(tloopstart))  # TIMING
        # end
    end

    # println("--> Time computing velocity for field point: ", elapsed_time(twhilestart))  # TIMING

    return rtn_vel
end