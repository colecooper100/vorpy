module weighted_biot_savart_integrator

using Base.Threads
using StaticArrays
using LinearAlgebra


export wbs_cpu  # wbs_1fp, INTGR_RTN_TYP,


###############################################################
# Bring in the utility functions
include("./wbs_utility_fns.jl")

# Set the weight function used by the WBS integrand
include("./weight_functions/bernstein_polynomial_weight.jl")
function weight_function(delta::T) where {T<:AbstractFloat}
    return bernstein_polynomial_weight(delta)
end

# Set the INTGRTNTYP to the return type of the 
# integrand. This is used by other functions.
# INTGR_RTN_TYP{T} = SVector{3, T}
# Set the integrand function to the WBS integrand
# include("./integrand_functions/vel_velgrad_integrand.jl")
include("./integrand_functions/wbs_integrand_function.jl")
# Set the return type of the integrand function
function integrand(
            params::SVector{13, T},
            ell::T) where {T<:AbstractFloat}
    # rtnvec, rtngrad, endofseg = vel_velgrad_integrand(fp, vpprops, ell)
    return wbs_integrand_function(params, ell)
end

# Set the vortex properties interpolator
include("./path_interpolators/linear_polygonal_path.jl")
include("./vortex_interpolators/piecewise_linear_vortex.jl")
function vortex_interpolator(
                        vppI::SVector{3, T},
                        vppF::SVector{3, T},
                        cradI::T,
                        cradF::T,
                        circI::T,
                        circF::T,
                        ell::T) where {T<:AbstractFloat}
    
    vpp, unttanvpp, elltrue, endofseg = linear_polygonal_path(vppI, vppF, ell)
    crad, circ = piecewise_linear_vortex(vpp, vppI, vppF, cradI, cradF, circI, circF)
    return vpp, unttanvpp, crad, circ, elltrue, endofseg
end

# Set integration method
include("integration_methods/nonuniform_trapezoidal_rule.jl")
function integrator(
            stepsize::T,    
            params::SVector{13, T}) where {T<:AbstractFloat}
    return nonuniform_trapezoidal_rule(stepsize, params)
end


###############################################################
function wbs_1seg(
                fp::SVector{3, T},
                vpps::AbstractArray{T, 2},
                crads::AbstractArray{T, 1},
                circs::AbstractArray{T, 1},
                stepscalar::T,
                minstepsize::T,
                segindx::Integer) where {T<:AbstractFloat}

    # Get vortex segment
    # vorseg := (vpp1, vpp2, crad1, crad2, circ1, circ2)
    seg = packseg(vpps, crads, circs, segindx)

    # Compute step size for the integrator
    stepsize = compstepsize(seg[7], seg[8], stepscalar, minstepsize)

    # Make params vector for integrator
    params = SVector{13, T}(fp..., seg...)

    # Integrate WBS integrand for the current
    # segment
    segvals = integrator(stepsize, params)

    return segvals
end


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

I type the arguments as AbstractArrays because
vpps, crads, cirds is generally an Array or CUDA array 
===================================================#
function wbs_1fp(
                fp::SVector{3, T},
                vpps::AbstractArray{T, 2},
                crads::AbstractArray{T, 1},
                circs::AbstractArray{T, 1},
                # # stepscalar: step size scalar for the integrator
                # # The integrator's step size is defined as
                # # stepsize = stepscalar * min(core_radius)
                stepscalar::T,
                minstepsize::T
                ) where {T<:AbstractFloat}

    # Compute the number of vortex path segments
    numpathsegs = size(vpps, 2) - 1

    # Initialize the return array
    rtnvals = SVector{3, T}(0, 0, 0)

    for segindx in 1:numpathsegs
        # Accumulate the result for all segments
        rtnvals = rtnvals .+ wbs_1seg(fp, vpps, crads, circs, stepscalar, minstepsize, segindx)
    end

    return rtnvals
end

#===============================================
This function implements the CPU version of the
weighted Biot-Savart solver. The solver loops
over all the field points supplied to the
method and returns a vector of velocities 
equal in langth to the number of field
points supplied.
The method for computing the velocity at a
single field point given a vortex path is
implemented to run on the CPU or GPU. This,
file handles the parallelization of the method
(should any be available).
===============================================#
function wbs_cpu(
            fieldpoints,
            vorpathpoints,
            cordradii,
            circulations;
            stepsizescalar::T,
            minstepsize::T,
            threaded::Bool=false) where {T<:AbstractFloat}

    # println("typeof(fieldpoints): ", typeof(fieldpoints))
    # println("get3col(fieldpoints, 1): ", get3col(fieldpoints, 1))
    # println("Base.supertype(fieldpoints): ", Base.supertype(fieldpoints))
    # Create an array to store the return values
    # The first index will be equal to the length
    # of the return of the integrand function.
    rtnvals = zeros(T, 3, size(fieldpoints, 2))

    # Loop over the field points
    if threaded
        @threads for fpindx in axes(fieldpoints, 2)
            fp = get3col(fieldpoints, fpindx)
            rtnvals[:, fpindx] .= wbs_1fp(
                                                fp,
                                                vorpathpoints,
                                                cordradii,
                                                circulations,
                                                stepsizescalar,
                                                minstepsize)
                            
        end
    else
        for fpindx in axes(fieldpoints, 2)
            fp = get3col(fieldpoints, fpindx)
            rtnvals[:, fpindx] .= wbs_1fp(
                                                fp,
                                                vorpathpoints,
                                                cordradii,
                                                circulations,
                                                stepsizescalar,
                                                minstepsize)

        end
    end

    return rtnvals
end

end # module weighted_biot_savart_solver
