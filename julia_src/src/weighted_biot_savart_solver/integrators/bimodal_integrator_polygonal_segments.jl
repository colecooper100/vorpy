using StaticArrays: SVector
using LinearAlgebra: norm, cross


###### Set integration method ######
# Used to determine what part of the segment is
# inside the cutoff radius
include("polygonal_line_segmenter.jl")
# Used when inside the cutoff radius
include("nonuniform_trapezoidal_rule.jl")


###### Functions ######
# Analytical solution for a polygonal model
# See "Compact expressions for the Biot-Savart fields of a filamentary segment"
# by James D. Hanson and Steven P. Hirshman
function analytical_solution_polygonal_model(fp, vpp1, vpp2, cir1, cir2)
    # Compute xi for the end points of the path
    # segment
    xi1 = vpp1 .- fp
    xi2 = vpp2 .- fp
    ximag1 = norm(xi1)
    ximag2 = norm(xi2)

    # println("Inside analytical_solution_polygonal_model...")  # DEBUG
    seglen = norm(vpp2 .- vpp1)
    # println("seglen: ", seglen)  # DEBUG
    segtan = (vpp2 - vpp1) / seglen
    # println("segtan: ", segtan)  # DEBUG
    # The paper has the following convention
    # \vec \xi_2 = \vec \xi_1 - \vec \ell
    # where \vec \ell = \vec \xi_1 - \vec \xi_2
    # which is the opposite to my convention thus
    # the cross product is multiplied by -1
    direc = -cross(segtan, xi1)
    # println("direc: ", direc)  # DEBUG
    epsilon = seglen / (ximag1 + ximag2)
    term1 = 2 * epsilon / (1 - epsilon^2)
    term2 = 1 / (ximag1 * ximag2)
    avgcir = (cir1 + cir2) / 2
    # println("avgcir: ", avgcir)  # DEBUG
    return avgcir .* term1 .* term2 .* direc ./ (4 * Float32(pi))
end


###### Integrate along a given path segment ######
#==============================================
The bimodal integrator numerically solves the
Biot-Savart law when xi is within some cutoff
radius. Outside of the cutoff, the analytical
solution for a polygonal model is used.
==============================================#
function bimodal_integrator_segment(stepsize, fp, vpp1, vpp2, vcr1, vcr2, cir1, cir2; verbose=false)
    if verbose
        println("Inside bimodal_segment_integrator...")
    end

    # Set the cutoff radius as some multiple of
    # the maximum of the core radii
    cutoffrad = Float32(5) * max(vcr1, vcr2)

    if verbose
        println("* cutoffrad: ", cutoffrad)
    end

    # Split the path segment into parts that are
    # inside and outside the cutoff radius
    vppincutoff, vppoutcutoff = polygonal_line_segmenter(vpp1, vpp2, fp, cutoffrad)

    # println("fp: ", fp)  # DEBUG
    # println("vppincutoff: ", vppincutoff)  # DEBUG
    # println("vppoutcutoff: ", vppoutcutoff)  # DEBUG

    velincutoff = 0
    if vppincutoff !== nothing
        velincutoff = nonuniform_trapezoidal_rule_segment(stepsize, fp, vppincutoff[:, 1], vppincutoff[:, 2], vcr1, vcr2, cir1, cir2)
    end

    veloutcutoff = 0
    if vppoutcutoff !== nothing
        for seg in vppoutcutoff
            veloutcutoff = veloutcutoff .+ analytical_solution_polygonal_model(fp, seg[:, 1], seg[:, 2], cir1, cir2)
        end
    end

    # println("velincutoff: ", velincutoff)  # DEBUG
    # println("veloutcutoff: ", veloutcutoff)  # DEBUG

    return velincutoff .+ veloutcutoff
end
