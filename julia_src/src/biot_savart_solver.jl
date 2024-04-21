using StaticArrays: SVector
using LinearAlgebra: norm, cross


###### Set BS integrator ######
# println("Inside biot_savart_solver_vortex_line.jl... ", "PWD: ", pwd())  # DEBUG
include("biot_savart_segment_integrator.jl")
bs_integrator(stepsize, fp, vpp1, vpp2, vcr1, vcr2, cir1, cir2) = bs_nonuniform_trapezoidal_rule_segment(stepsize, fp, vpp1, vpp2, vcr1, vcr2, cir1, cir2)


###### Utility functions ######
# Function for getting a vortex path segment
# which works on the GPU
function get_segment(vpps, cdms, circs, indx)
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
function velocity_at_field_point_bs(fp, vpps, cdms, circs)
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
        seg = get_segment(vpps, cdms, circs, segindx)
        # Rough estimate of the average core diameter
        # for the current segment
        avg_cdm = sum(seg[3:4]) / Float32(2)
        # The stepsize of the integrator is a
        # fraction of the average core diameter of
        # the segment.
        # THE stepsize HAS A SIGNIFICANT IMPACT ON
        # PERFORMANCE (OBVIOUSLY). I HAVE FOUND
        # THAT A stepsize OF 1 CORE DIAMETER IS THE
        # LARGEST THAT CAN BE USED WITHOUT NEGATIVELY
        # AFFECTING THE ACCURACY OF THE SOLUTION.
        stepsize = Float32(1) * avg_cdm
        segvel = bs_integrator(stepsize, fp, seg...)
        rtn_vel = rtn_vel .+ segvel
        segindx += UInt32(1)  # Advance the loop counter
    end

    return rtn_vel
end