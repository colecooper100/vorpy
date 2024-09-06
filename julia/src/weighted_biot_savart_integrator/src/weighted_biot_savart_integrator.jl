module weighted_biot_savart_integrator

using Base.Threads
using StaticArrays
using LinearAlgebra


export wbs_cpu


###############################################################
# Function for picking a field point out of a
# 3xN array of field points
function getfp(fps::AbstractArray{T}, indx::Integer)::SVector{3, T} where {T<:AbstractFloat}
    return SVector{3, T}(fps[1, indx], fps[2, indx], fps[3, indx])
end

function compstepsize(
                    cradI::T,
                    cradF::T,
                    stepscalar::T,
                    minstepsize::T)::T where {T<:AbstractFloat}

    mincradstep = stepscalar * T(min(cradI, cradF))
    stepsize = max(mincradstep, minstepsize)

    return stepsize
end

function pckprams(
    fp::SVector{3, T},
    vpps::AbstractArray{T, 2},
    crads::AbstractArray{T, 1},
    circs::AbstractArray{T, 1},
    indx::Integer)::SVector{13, T} where {T<:AbstractFloat}

    # Get starting point of segment
    vppI = SVector{3, T}(
    vpps[1, indx],
    vpps[2, indx],
    vpps[3, indx])

    # Get the ending point of segment
    vppF = SVector{3, T}(
    vpps[1, indx+1],
    vpps[2, indx+1],
    vpps[3, indx+1])

    # @inbounds return vpp1, vpp2, crads[indx], crads[indx+1], circs[indx], circs[indx+1]
    return SVector{13, T}(fp..., vppI..., vppF..., crads[indx], crads[indx+1], circs[indx], circs[indx+1])
    # return SVector{10, T}(vppI..., vppF..., crads[indx], crads[indx+1], circs[indx], circs[indx+1])
end

# Set WBS integration method
include("integration_methods/nonuniform_trapezoidal_rule.jl")
# include("integration_methods/bimodal_polygonal_path_integrator/bimodal_polygonal_path_integrator.jl")
function WBSintegrator(
            stepsize::T,    
            params::SVector{13, T}) where {T<:AbstractFloat}
    sol, itercnt = nonuniform_trapezoidal_rule(stepsize, params)
    # sol, itercnt = bimodal_polygonal_path(stepsize, params)
    return sol
end


###############################################################
# function wbs_1seg(
#                 fp::SVector{3, T},
#                 # seg := SVector{10, T}(vppI..., vppF..., crads[indx], crads[indx+1], circs[indx], circs[indx+1])
#                 seg::SVector{10, T},
#                 stepscalar::T,
#                 minstepsize::T) where {T<:AbstractFloat}

#     # Compute step size for the integrator
#     stepsize = compstepsize(seg[7], seg[8], stepscalar, minstepsize)

#     # Integrate WBS integrand for the current
#     # segment
#     segvals = integrator(stepsize, params)

#     return segvals
# end


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
    # This should be the same length as the return
    # of the integrand function
    rtnvals = SVector{3, T}(0, 0, 0)

    for segindx in 1:numpathsegs
        # Pack params vector for integrator
        # params = SVector{13, T}(fp..., vppI..., vppF..., crads[indx], crads[indx+1], circs[indx], circs[indx+1])
        params = pckprams(fp, vpps, crads, circs, segindx)

        # Compute step size for the integrator
        stepsize = compstepsize(params[10], params[11], stepscalar, minstepsize)

        # Integrate WBS integrand for the current
        # segment and accumulate the result for all segments
        # segvel = wbs_1seg(fp, seg, stepscalar, minstepsize)
        # println("segvel: ", segvel)  # DEBUG
        # rtnvals = rtnvals .+ segvel
        rtnvals = rtnvals .+ WBSintegrator(stepsize, params)
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
            threaded::Bool) where {T<:AbstractFloat}

    # Create an array to store the return values
    # The first index will be equal to the length
    # of the return of the integrand function.
    rtnvals = zeros(T, 3, size(fieldpoints, 2))

    # Loop over the field points
    if threaded
        @threads for fpindx in axes(fieldpoints, 2)
            fp = getfp(fieldpoints, fpindx)
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
            fp = getfp(fieldpoints, fpindx)
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
