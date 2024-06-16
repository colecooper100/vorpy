###### Import modules and local scrips ######
using StaticArrays: SVector
using LinearAlgebra: norm, cross

# Utility function
# Used to determine what part of the segment is
# inside the cutoff radius
include("polygonal_line_segmenter.jl")


# Set integration method
# Numerical integator used when inside the
# cutoff radius
include(string(WEIGHTED_BIOT_SAVART_INTEGRATOR_METHODS, "/nonuniform_trapezoidal_rule/biot_savart_nonuniform_trapezoidal_rule.jl"))


###### Functions ######
# Analytical solution for a polygonal model
# See "Compact expressions for the Biot-Savart fields of a filamentary segment"
# by James D. Hanson and Steven P. Hirshman
function _analytical_solution_polygonal_model(fp::SVector{3, T},
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
function bimodal_biot_savart_integrator_polygonal_segments(stepsize::T,
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
    pathsegincutoff, pathsegoutcutoff = polygonal_line_segmenter(cutoffrad, fp, vpp1, vpp2)

    # Compute the velocity caused by the path segment
    # inside the cutoff radius
    velincutoff = SVector{3, T}(0, 0 ,0)
    if pathsegincutoff[1]
        velincutoff = biot_savart_nonuniform_trapezoidal_rule(stepsize, fp, pathsegincutoff[2][1], pathsegincutoff[2][2], crad1, crad2, circ1, circ2)
    end

    # Compute the velocity caused by the path segment
    # outside the cutoff radius
    veloutcutoff = SVector{3, T}(0, 0 ,0)
    for elm in pathsegoutcutoff
        if elm[1]
            velincutoff = velincutoff .+ _analytical_solution_polygonal_model(fp, elm[2][1], elm[2][2], circ1, circ2)
        end
    end

    return velincutoff .+ veloutcutoff
end
