#===============================================
Note: The following include statements are
specific to this script, thus they are included
here rather than in the main script (i.e.,
weighted_biot_savart_integrator.jl).
===============================================#
# Numerical integration method used when inside
# the cutoff radius.
# **This is specific to this script and not used
# by anything else, so keep this include
# statement here**
include("wbs_nonuniform_trapezoidal_rule.jl")

# Utility function which determines what part of
# the segment is inside the cutoff radius.
# **This is specific to this script and not used
# by anything else, so keep this include
# statement here**
include("polygonal_line_segmenter.jl")


# Analytical solution for a polygonal model
# See "Compact expressions for the Biot-Savart fields of a filamentary segment"
# by James D. Hanson and Steven P. Hirshman
function velocity_polygonal_path_analytical(fp::SVector{3, T},
                                                vpp1::SVector{3, T},
                                                vpp2::SVector{3, T},
                                                circ1::T,
                                                circ2::T) where {T<:AbstractFloat}
    # Compute xi for the end points of the path
    # segment
    xi1 = vpp1 .- fp
    xi2 = vpp2 .- fp
    ximag1 = norm(xi1)
    ximag2 = norm(xi2)
    seglen = norm(vpp2 .- vpp1)
    segtan = (vpp2 - vpp1) / seglen

    # The paper has the following convention
    # \vec \xi_2 = \vec \xi_1 - \vec \ell
    # where \vec \ell = \vec \xi_1 - \vec \xi_2
    # which is the opposite to my convention thus
    # the cross product is multiplied by -1
    direc = -cross(segtan, xi1)
    epsilon = seglen / (ximag1 + ximag2)
    term1 = 2 * epsilon / (1 - epsilon^2)
    term2 = 1 / (ximag1 * ximag2)
    avgcir = (circ1 + circ2) / 2

    return avgcir .* term1 .* term2 .* direc ./ (4 * T(pi))
end


###### Integrate along a given path segment ######
#==============================================
The bimodal integrator numerically solves the
Biot-Savart law when xi is within some cutoff
radius. Outside of the cutoff, the analytical
solution for a polygonal model is used.
==============================================#
function bimodal_integrator_polygonal_path(stepsize::T,
                                                    fp::SVector{3, T},
                                                    vpp1::SVector{3, T},
                                                    vpp2::SVector{3, T},
                                                    crad1::T,
                                                    crad2::T,
                                                    circ1::T,
                                                    circ2::T) where {T<:AbstractFloat}
    # Set the cutoff radius as some multiple of
    # the maximum of the core radii
    cutoffrad = T(5) * max(crad1, crad2)
    # println("cutoffrad = ", cutoffrad)  # DEBUG

    # Split the path segment into parts that are
    # inside and outside the cutoff radius
    # In order to get things to work with the GPU
    # the return of this function had to be the same
    # (as in number of values returned and their types)
    # regardless of if any part of the path is inside
    # the cutoff radius.
    # The return has the following format:
    # pathsegincutoff = (Bool, (Array{SVector{3, T}, 1}, Array{SVector{3, T}, 1}))
    # pathsegoutcutoff = ((Bool, (Array{SVector{3, T}, 1}, Array{SVector{3, T}, 1})), (Bool, (Array{SVector{3, T}, 1}, Array{SVector{3, T}, 1})))
    # The boolean value indicates if the path segment is
    # used or not (e.g., if parth of the path is in
    # the cutoff radius, then the boolean value in
    # pathsegincutoff will be true). pathsegoutcutoff
    # can have up to two path segments, which is why
    # there are two parts to the tuple. 
    pathincutoff, pathoutcutoff = polygonal_line_segmenter(cutoffrad, fp, vpp1, vpp2)
    # println("pathincutoff = ", pathincutoff)  # DEBUG
    # println("pathoutcutoff = ", pathoutcutoff)  # DEBUG

    # Compute the velocity caused by the path segment
    # inside the cutoff radius
    velincutoff = SVector{3, T}(0, 0 ,0)
    if pathincutoff[1]
        # This should not be set to the global
        # function wbs_integrator or the code will
        # not terminate (you get a buffer overflow
        # error).  
        velincutoff = wbs_nonuniform_trapezoidal_rule(stepsize, fp, pathincutoff[2][1], pathincutoff[2][2], crad1, crad2, circ1, circ2)
    end

    # Compute the velocity caused by the path segment
    # outside the cutoff radius
    veloutcutoff = SVector{3, T}(0, 0 ,0)
    for elm in pathoutcutoff
        if elm[1]
            veloutcutoff = veloutcutoff .+ velocity_polygonal_path_analytical(fp, elm[2][1], elm[2][2], circ1, circ2)
        end
    end

    # println("velincutoff = ", velincutoff)  # DEBUG
    # println("typeof(velincutoff) = ", typeof(velincutoff))  # DEBUG
    # println("veloutcutoff = ", veloutcutoff)  # DEBUG
    # println("typeof(veloutcutoff) = ", typeof(veloutcutoff))  # DEBUG
    # println("velincutoff .+ veloutcutoff = ", velincutoff .+ veloutcutoff)  # DEBUG
    # println("typeof(velincutoff .+ veloutcutoff) = ", typeof(velincutoff .+ veloutcutoff))  # DEBUG

    return velincutoff .+ veloutcutoff
end
