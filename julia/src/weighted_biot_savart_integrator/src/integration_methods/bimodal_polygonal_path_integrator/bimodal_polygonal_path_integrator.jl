

# Set the method of numerical integration to
# use when inside the cutoff radius (i.e.,
# the numerical integration method for solving
# the weighted Biot-Savart law).
include("../nonuniform_trapezoidal_rule.jl")

# # Utility function spliting the path up into
# # The segments inside the cutoff radius and
# # outside.
# # include("polygonal_line_segmenters_complex.jl")
# include("polygonal_line_segmenter_simple.jl")


##############################################################

# function pckprams(
#     fp::SVector{3, T},
#     vpps::AbstractArray{T, 2},
#     crads::AbstractArray{T, 1},
#     circs::AbstractArray{T, 1},
#     indx::Integer)::SVector{13, T} where {T<:AbstractFloat}

#     # Get starting point of segment
#     vppI = SVector{3, T}(
#     vpps[1, indx],
#     vpps[2, indx],
#     vpps[3, indx])

#     # Get the ending point of segment
#     vppF = SVector{3, T}(
#     vpps[1, indx+1],
#     vpps[2, indx+1],
#     vpps[3, indx+1])

#     # @inbounds return vpp1, vpp2, crads[indx], crads[indx+1], circs[indx], circs[indx+1]
#     return SVector{13, T}(fp..., vppI..., vppF..., crads[indx], crads[indx+1], circs[indx], circs[indx+1])
#     # return SVector{10, T}(vppI..., vppF..., crads[indx], crads[indx+1], circs[indx], circs[indx+1])
# end


# Analytical solution for a polygonal model
# See "Compact expressions for the Biot-Savart fields of a filamentary segment"
# by James D. Hanson and Steven P. Hirshman
function velocity_polygonal_line(fp::SVector{3, T},
                                                vpp1::SVector{3, T},
                                                vpp2::SVector{3, T},
                                                circ::T) where {T<:AbstractFloat}
    # Compute xi for the end points of the path
    # segment
    RIvec = vpp1 .- fp
    RFvec = vpp2 .- fp
    RImag = norm(RIvec)
    RFmag = norm(RFvec)
    segvec = vpp2 .- vpp1
    seglen = norm(segvec)
    segtan = segvec ./ seglen

    # The paper has the following convention
    # \vec R_F = \vec R_I - \vec \ell
    # where \vec \ell = \vec R_I - \vec R_F
    # which is the opposite to my convention thus
    # the cross product is multiplied by -1
    dir = -cross(segtan, RIvec)
    epsilon = seglen / (RImag + RFmag)
    term1 = epsilon / (1 - epsilon^2)
    term2 = 1 / (RImag * RFmag)

    return (circ / (2 * T(pi))) .* term1 .* term2 .* dir
end


###### Integrate along a given path segment ######
#==============================================
The bimodal integrator numerically solves the
Biot-Savart law when xi is within some cutoff
radius. Outside of the cutoff, the analytical
solution for a polygonal model is used.
==============================================#
function bimodal_polygonal_path(
                            stepsize::T,
                            # params = SVector{13, T}(fp..., vppI..., vppF..., crads[indx], crads[indx+1], circs[indx], circs[indx+1])   
                            params::SVector{13, T}) where {T<:AbstractFloat}
   
    # Unpack the parameters
    fp = SVector{3, T}(params[1], params[2], params[3])
    vppI = SVector{3, T}(params[4], params[5], params[6])
    vppF = SVector{3, T}(params[7], params[8], params[9])
    crad1 = params[10]
    crad2 = params[11]
    circI = params[12]

    # Set the cutoff radius as some multiple of
    # the maximum of the core radii
    cutoffrad = T(5) * max(crad1, crad2)

    segvec = vppF .- vppI
    seglen = norm(segvec)
    segtan = segvec ./ seglen

    RIvec = vppI .- fp
    RFvec = vppF .- fp
    RImag = norm(RIvec)
    RFmag = norm(RFvec)
    RIpar = dot(RIvec, segtan)
    RIper = sqrt(RImag^2 - RIpar^2)

    if RImag <= cutoffrad || RFmag <= cutoffrad || (RIpar > 0 && RIper <= cutoffrad)
        # Inside cutoff radius
        # Send to WBS integrator
        #======================================
        function nonuniform_trapezoidal_rule(
                                    stepsize::T,
                                    params::SVector) where {T<:AbstractFloat}

            return rtnval, itercount
        ======================================#
        rtnvals, itercnt = nonuniform_trapezoidal_rule(stepsize, params)
    else
        # Outside cutoff radius
        # Use analytical solution
        #======================================
        function velocity_polygonal_line(fp::SVector{3, T},
                                                vpp1::SVector{3, T},
                                                vpp2::SVector{3, T},
                                                circ::T) where {T<:AbstractFloat}
        ======================================#
        rtnvals = velocity_polygonal_line(fp, vppI, vppF, circI)
        itercnt = T(0)
    end

    return rtnvals, itercnt
end
