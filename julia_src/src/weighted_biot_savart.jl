using StaticArrays: SVector
using LinearAlgebra: norm, cross


###### Set BS integrator ######
# println("Inside biot_savart_solver_vortex_line.jl... ", "PWD: ", pwd())  # DEBUG
include("biot_savart_segment_integrator.jl")
bs_integrator(stepsize, fp, vpp1, vpp2, vcr1, vcr2, cir1, cir2) = bs_nonuniform_trapezoidal_rule_segment(stepsize, fp, vpp1, vpp2, vcr1, vcr2, cir1, cir2)


###### Utility functions ######
# Function for getting a vortex path segment
# which works on the GPU
function _getseg(vpps, cdms, circs, indx)
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

    @inbounds return vpp1, vpp2, cdms[indx], cdms[indx+1], circs[indx], circs[indx+1]
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
function _weighted_biot_savart(fp, vpps, cdms, circs; verbose=false)
    # Compute the number of vortex path segments
    num_vsegs = UInt32(size(vpps, 2)) - UInt32(1)

    # Initialize the return velocity
    rtn_vel = SVector{3, Float32}(0, 0, 0)

    # Step through each vortex path segment.
    # We are using a while loop because the CUDA.jl
    # docs says this is more efficient than using a
    # for loop with a step interval.
    # I WOULD LIKE TO KNOW IF THIS IS TRUE!
    # TIMING
    if verbose
        include("/home/user1/Dropbox/code/vorpy/julia_src/src/timing.jl")
        println("Inside _weighted_biot_savart...")
        println("* Looping over all vortex path segments...")
        # Initial time for entire execution
        t0 = time_ns()
    end
    segindx = UInt32(1)
    while segindx <= num_vsegs
        # Get segment
        seg = _getseg(vpps, cdms, circs, segindx)

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
        STEP_SIZE_SCALAR = Float32(5)
        avg_cdm = sum(seg[3:4]) / Float32(2)
        stepsize = STEP_SIZE_SCALAR * avg_cdm
        # TIMING
        if verbose
            print("  * Segment: ", segindx, "/", num_vsegs, ", ")
            print("integrator step size:", stepsize,  "... ")
            t1 = time_ns()
        end
        segvel = bs_integrator(stepsize, fp, seg...)
        # TIMING
        if verbose
            println("Done ", elapsed_time(t1), " seconds.")
        end
        rtn_vel = rtn_vel .+ segvel
        segindx += UInt32(1)  # Advance the loop counter
    end

    # TIMING
    if verbose
        println("* Total elapsed time: ", elapsed_time(t0), " seconds")
        println("Leaving _weighted_biot_savart.")
    end

    return rtn_vel
end