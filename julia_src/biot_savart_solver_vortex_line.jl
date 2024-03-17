using StaticArrays: SVector
using LinearAlgebra: norm, cross

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

#===================================================
We compute the flow velocity at a field point using
the Biot-Savart law. The vortex path is defined by a
series of path points. Additionally, at each point
we define the core radus, and circulation.

This function loops through each segment of the
vortex path summing the velocity contribution of
each path segment. Specifically, the function
computes a stepsize for the integrator, it then
passes the step size, field point, and segment to
the integrator, which returns the velocity for that
segment.
===================================================#
function velocity_field_point(fp, vpps, vcrs, cirs)
    # Compute the number of vortex path segments
    num_vsegs = UInt32(size(vpps, 2)) - UInt32(1)

    # Set the initial flow velocity
    velocity = SVector{3, Float32}(0, 0, 0)

    # Step through each vortex path segment.
    # We are using a while loop because the CUDA.jl
    # docs says this is more efficient than using a
    # for loop with a step interval.
    segindx = UInt32(1)
    while segindx <= num_vsegs
        # The stepsize of the integrator is some
        # fraction of the average core diameter of
        # the segment.
        # THE stepsize HAS A SIGNIFICANT IMPACT ON
        # PERFORMANCE (OBVIOUSLY). I HAVE FOUND
        # THAT A stepsize OF 1 IS THE LARGEST THAT
        # CAN BE USED WITHOUT NEGATIVELY AFFECTING
        # THE ACCURACY OF THE SOLUTION.
        stepsize = (vcrs[segindx] + vcrs[segindx+1]) / Float32(2) / Float32(1)
        segvel = bs_integrator(stepsize, fp, get_segment(vpps, vcrs, cirs, segindx)...)
        velocity = velocity .+ segvel
        segindx += UInt32(1)  # Advance the loop counter
    end

    return velocity
end