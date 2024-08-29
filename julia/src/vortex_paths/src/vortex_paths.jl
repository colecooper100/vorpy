module vortex_paths

using StaticArrays
using LinearAlgebra

export linevortex,
        u_inflong_line,
        u_seg_polyline,
        u_polyline,
        ringvortex#,
        # gradu_polylineseg


function linevortex(
                numsegs::Integer,
                path_start::AbstractArray{T, 1},
                path_end::AbstractArray{T, 1})::AbstractArray{T, 2} where {T<:AbstractFloat}
    # num_vpps_segs needs to be >= 1
    if numsegs >= 1
        vorpath = stack(range(path_start, path_end, length=numsegs+1), dims=2)
    else
        throw("numsegs must to be an Integer subtype type and equal to or greater than 1")
    end
    # return SMatrix{size(vorpathraw)..., T2}(vorpathraw)
    return vorpath
end

# Analytical velocity for LineVortex
# For a Lamb-Oseen vortex, with cord radius a = \sqrt{2 \nu t}
# the vorticity in cylindrical coordinates
# (s, \phi, z) is given by:
# \vec \omega (\vec r, t) = (\Gamma / 2 \pi a^2) \exp(-s^2 / 2 a^2)  \hat z
# The velocity field is given by:
# \vec u (\vec r, t) = (\Gamma / 2 \pi s) (1 - \exp(-s^2 / 2 a^2)) \hat \phi
function u_inflong_line(
                        fp::AbstractArray{T, 1},
                        vppI::AbstractArray{T, 1},
                        vppF::AbstractArray{T, 1},
                        crad::T,
                        circ::T)::AbstractArray{T, 1} where {T<:AbstractFloat}  
    tvec = vppF .- vppI
    tmag = norm(tvec)
    tdir = tvec ./ tmag

    R1 = fp .- vppI
    R1para = dot(R1, tdir) .* tdir
    R1perp = R1 .- R1para
    R1perpmag = norm(R1perp)

    veldir = cross(tdir, R1perp ./ R1perpmag)

    rtnvelmag = zeros(T, 3)
    # If crad=T(0), then we assume an 
    # infinitesimally thin core where
    #\vec u = \Gamma / 2 \pi s) \hat \phi
    if crad == T(0)
        rtnvelmag .= circ / (2 * pi * R1perpmag)
    else
        rtnvelmag .= circ / (2 * pi * R1perpmag) * (1 - exp(-R1perpmag^2 / (2 * crad^2)))
    end
    
    return rtnvelmag .* veldir
end

# Analytical velocity for a polygonal segment
function u_seg_polyline(
                    fp::AbstractArray{T},
                    vppI::AbstractArray{T},
                    vppF::AbstractArray{T},
                    circ::T) where {T<:AbstractFloat}
    segvec = vppF .- vppI
    segmag = norm(segvec)
    untsegtan = segvec ./ segmag
    # println("fp: ", fp)  # DEBUG
    # println("vppI: ", vppI)  # DEBUG
    RIvec = fp .- vppI
    RImag = norm(RIvec)
    RFvec = fp .- vppF
    RFmag = norm(RFvec)

    epsilon = segmag / (RImag + RFmag)
    dir = cross(untsegtan, RIvec)

    trm1 = circ / (2 * pi)
    trm2 = epsilon / (1 - epsilon^2)
    trm3 = 1 / (RImag * RFmag)

    return (trm1 * trm2 * trm3) .* dir
end

function u_polyline(
                fp::AbstractArray{T},
                vpps::AbstractArray{T},
                circs::AbstractArray{T}) where {T<:AbstractFloat}

    numsegs = size(vpps, 2) - 1
    rtnvel = zeros(T, 3)
    for i in 1:numsegs
        vppI = vpps[:, i]
        vppF = vpps[:, i+1]
        circ = circs[i]
        #================================
        u_seg_polyline(
                        fp::AbstractArray{T, 1},
                        vppI::AbstractArray{T, 1},
                        vppF::AbstractArray{T, 1},
                        circ::T)::AbstractArray{T, 1} where {T<:AbstractFloat}
        ================================#   
        rtnvel .+= u_seg_polyline(fp, vppI, vppF, circ)
    end
    return rtnvel
end

# # Analytical velocity gradient for a polygonal segment
# function gradu_polylineseg(
#                     fp::AbstractArray{T, 1},
#                     vppI::AbstractArray{T, 1},
#                     vppF::AbstractArray{T, 1},
#                     circ::T)::AbstractArray{T, 2} where {T<:AbstractFloat}
    
#     segvec = vppF .- vppI
#     segmag = norm(segvec)
#     untsegtan = segvec ./ segmag
#     RIvec = fp .- vppI
#     RImag = norm(RIvec)
#     untRIvec = RIvec ./ RImag
#     RFvec = fp .- vppF
#     RFmag = norm(RFvec)
#     untRFvec = RFvec ./ RFmag
#     epsilon = segmag / (RImag + RFmag)

#     if (RImag + RFmag - segmag) < (5 * eps(T))
#         println("Warning: Result divergent, returning zeros")
#         return zeros(T, 3, 3)
#     else
#         dir = cross(untsegtan, RIvec)

#         numer = (1 + epsilon^2)
#         denom = (1 - epsilon^2)

#         qtrm1 = numer / denom
#         qtrm2 = (untRIvec .+ untRFvec) ./ (RImag + RFmag)
#         qtrm3 = (untRIvec ./ RImag) .+ (untRFvec ./ RFmag)
#         q = (qtrm1 .* qtrm2) .+ qtrm3

#         Imat = SMatrix{3, 3, T}(1, 0, 0, 0, 1, 0, 0, 0, 1)
#         trm1 = -circ / (2 * pi)
#         trm2 = epsilon / (denom * RImag * RFmag)
#         trm4 = q * transpose(dir)
#         trm5 = cross(Imat, untsegtan)

#         return (trm1 * trm2) .* (trm4 .+ trm5)
#     end
# end


function ringvortex(numsegs::T1, radius::T2)::AbstractArray{T, 2} where {T1<:Integer, T2<:AbstractFloat}
    # num_vpps_segs = 8
    # radius = 1.0
    # phase = pi/8
    vorpathraw = zeros(T, 3, numsegs+1)
    vorpathraw[1, :] .= radius .* cos.(range(0, stop=2*pi, length=numsegs+1))
    vorpathraw[2, :] .= radius .* sin.(range(0, stop=2*pi, length=numsegs+1))
    vorpathraw[:, end] .= vorpathraw[:, 1]  # Close the path
    return SMatrix{size(vorpathraw)..., T2}(vorpathraw)
end

# function u_ringvortex()::AbstractArray{T, 1}
# end


# function threefoldvortex()::AbstractArray{T, 2}
# end

# function trefoilvortex()::AbstractArray{T, 2}
# end

end # module vortex_paths
