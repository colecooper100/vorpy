module vortex_paths

using StaticArrays
using LinearAlgebra

export LineVortex, u_LineVortex, RingVortex

#============== LineVortex ==============#
function LineVortex(numsegs::T1, path_start::AbstractArray{T2, 1}, path_end::AbstractArray{T2, 1}) where {T1<:Integer, T2<:AbstractFloat}
    # num_vpps_segs needs to be >= 1
    if numsegs >= 1
        vorpath = stack(range(path_start, path_end, length=numsegs+1), dims=2)
    else
        throw("numsegs needs to be an Int type and greater than or equal to 1")
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
function u_LineVortex(fp::AbstractArray{T, 1},
                        vppI::AbstractArray{T, 1},
                        vppF::AbstractArray{T, 1},
                        crad::T,
                        circ::T) where {T<:AbstractFloat}  
    # If crad=0.0, then we assume an 
    # infinitesimally thin core where
    #\vec u = \Gamma / 2 \pi s) \exp(-s^2 / 2 a^2) \hat \phi

    tvec = vppF .- vppI
    # println("tvec: ", tvec)  # DEBUG
    #============================================
    ** READ ME **
    I was getting an error when I used the norm
    function. This happened all of a sudden and
    it seems to work everywhere else, so I'm not
    sure what the issue is with this module.
    ============================================#
    tmag = norm(tvec)
    # tmag = sqrt(sum(tvec .* tvec)) 
    # println("tmag: ", tmag)  # DEBUG
    tdir = tvec ./ tmag

    R1 = fp .- vppI
    R1para = dot(R1, tdir) .* tdir
    R1perp = R1 .- R1para
    R1perpmag = norm(R1perp)
    # R1perpmag = sqrt(sum(R1perp .* R1perp))
    # println("R1perpmag: ", R1perpmag)

    veldir = cross(tdir, R1perp ./ R1perpmag)

    rtnvelmag = zeros(Float64, 3)
    if crad === nothing
        rtnvelmag .= circ / (2 * pi * R1perpmag)
    else
        rtnvelmag .= circ / (2 * pi * R1perpmag) * (1 - exp(-R1perpmag^2 / (2 * crad^2)))
    end
    
    return rtnvelmag .* veldir
end


#============== RingVortex ==============#
function RingVortex(numsegs::T1, radius::T2) where {T1<:Integer, T2<:AbstractFloat}
    # num_vpps_segs = 8
    # radius = 1.0
    # phase = pi/8
    vorpathraw = zeros(Float64, 3, numsegs+1)
    vorpathraw[1, :] .= radius .* cos.(range(0, stop=2*pi, length=numsegs+1))
    vorpathraw[2, :] .= radius .* sin.(range(0, stop=2*pi, length=numsegs+1))
    vorpathraw[:, end] .= vorpathraw[:, 1]  # Close the path
    return SMatrix{size(vorpathraw)..., T2}(vorpathraw)
end

end # module vortex_paths
