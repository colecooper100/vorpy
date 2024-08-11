module weighted_biot_savart_solver

using StaticArrays
using LinearAlgebra
using utilities

#========================================
I wanted to make this code modular, so
new methods and functions could be addded
or swapped out with minimal change to the
rest of the code.
It seemed logical to put all of the code
needed to integrate the Biot-Savart law
for a single field point into the same
directory. Additionally, to ensure that
the code was modular, I made general
but specific names for functions used
through this code. In a perfect world
I could assign a specific function to a
new variable and then use that
variable/function as a function. But
this does not work on the GPU. So I make
a function (wrapper) which calls the
specific function.
========================================#
# Set the weight function
include("bernstein_polynomial_weight.jl")
function weight_function(delta::T) where {T<:AbstractFloat}
    return bernstein_polynomial_weight(delta)
end

# Set the model for interpolating vortex
# properties (this was a originally callled
# `vortex_model`)
include("piecewise_linear_segments.jl")
function vortex_interpolator(ell::T,
                                vppI::SVector{3, T},
                                vppF::SVector{3, T},
                                cradI::T,
                                cradF::T,
                                circI::T,
                                circF::T) where {T<:AbstractFloat}
    return piecewise_linear_segments(ell, vppI, vppF, cradI, cradF, circI, circF)
end

# Set the WBS numerical integration method
# wbs_integration_method is passed
# (stepsize, fp, vpp1, vpp2, crad1, crad2, circ1, circ2)
include("bimodal_polygonal_path_integrator.jl")
function wbs_integration_method(stepsize::T,
                                fp::SVector{3, T},
                                vpp1::SVector{3, T},
                                vpp2::SVector{3, T},
                                crad1::T,
                                crad2::T,
                                circ1::T,
                                circ2::T) where {T<:AbstractFloat}
    return bimodal_integrator_polygonal_path(stepsize, fp, vpp1, vpp2, crad1, crad2, circ1, circ2)
end

export u_wbs_1fp


#===================================================
**Function for computing flow velocity at ONE field point**

We compute the flow velocity at ONE field point using
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
function u_wbs_1fp(fp::AbstractArray{T, 1},
    vpps::AbstractArray{T, 2},
    crads::AbstractArray{T, 1},
    circs::AbstractArray{T, 1},
    # stepscalar: step size scalar for the integrator
    # The integrator's step size is defined as
    # stepsize = stepscalar * min(core_radius)
    stepscalar::T)::SVector{3, T} where {T<:AbstractFloat}
    

    # Compute the number of vortex path segments
    num_vsegs = size(vpps, 2) - 1  # UInt32(size(vpps, 2)) - UInt32(1)
    # println("num_vsegs = ", num_vsegs)  # DEBUG

    # Initialize the return velocity
    rtn_vel = SVector{3, T}(0, 0, 0)

    # # t0 = time_ns()  # TIMING
    for segindx in 1:num_vsegs
        # Get segment
        # seg := (vpp1, vpp2, crad1, crad2, circ1, circ2)
        seg = getseg(vpps, crads, circs, segindx)
        # println("seg = ", seg)  # DEBUG

        # Determine an appropriate step size for the
        # integrator.
        # Get the minimum core diameter
        mincrad = min(seg[3], seg[4])
        stepsize = T(stepscalar * mincrad)
        # eps(<type>) returns the smallest positive
        # number that can be represented by the type.
        # (i.e. the machine epsilon or machine precision)
        # `stepsize` should always be positive, so we
        # assume this.
        if stepsize < (5 * eps(T))
            # `stepsizescalar` is $(stepsizescalar) and the minimum core radius is $(mincrad), this results in a step size of $(stepsize) which is smaller than the minimum allowable step size of $(5 * eps(T)) (note: this minimum is type specific)."
            throw(ArgumentError("The variable 'stepsizescalar' and/or the minimum core radius for some segment results in a integrator step size that is too small or negative."))
        end

        # t0 = time_ns()  # TIMING
        # Compute the velocity for the segment
        # segvel = bimodal_integrator_polygonal_path(stepsize, fp, seg...)  # DEBUG
        segvel = wbs_integration_method(stepsize, fp, seg...)
        # println("segvel = ", segvel)  # DEBUG
        # println("Time to compute segment velocity: ", (time_ns() - t0) / 1e9)  # TIMING

        # Add the segment velocity to the accumulated
        # velocity
        rtn_vel = rtn_vel .+ segvel
    end
    # println("Time to compute velocity at one field point: ", (time_ns() - t0) / 1e9)  # TIMING

    return rtn_vel
end

end # module weighted_biot_savart_solver
