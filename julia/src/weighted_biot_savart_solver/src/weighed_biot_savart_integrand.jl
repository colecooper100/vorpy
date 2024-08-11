# ###### Import modules and local scrips ######
# using StaticArrays: SVector, SMatrix
# using LinearAlgebra: norm, cross

# # Include the weight function
# include(string(WEIGHTED_BIOT_SAVART_INTEGRAND, "/bernstein_polynomial_weight.jl"))


# ###### Set the weight function ######
# function weight_function(delta)
#     return bernstein_polynomial_weight(delta)
# end


###### Integrand function ######
# The integrand always expects ell \in [0, seglen]
function weighted_biot_savart_integrand(ell::T,
                        fp::SVector{3, T},
                        vpp1::SVector{3, T},
                        vpp2::SVector{3, T},
                        crad1::T,
                        crad2::T,
                        circ1::T,
                        circ2::T) where {T<:AbstractFloat}
    # Get the vortex properties at the given
    # arc length ell
    vmodelprop = vortex_interpolator(ell, vpp1, vpp2, crad1, crad2, circ1, circ2)
    vpp = vmodelprop[1]
    vtan = vmodelprop[2]
    crad = vmodelprop[3]
    circ = vmodelprop[4]
    rtnell = vmodelprop[5]
    endofseg = vmodelprop[6]
    # println("typeof(endofseg): ", typeof(endofseg))  # DEBUG

    # Compute xi, the vector that extends from a
    # point on the vortex path to the field point
    xi = fp .- vpp
    ximag = norm(xi)

    # Initialize return values
    computed_wbs_integrand = SVector{3, T}(0, 0, 0)
    volgradtensor = SMatrix{3, 3, T}(0, 0, 0, 0, 0, 0, 0, 0, 0)

    # If ximag == 0 return zero for computed_wbs_integrand
    # and volgradtensor, otherwise do the calculation
    if ximag != 0
        vtanmag = norm(vtan)
        dir = cross(vtan, xi)
        weight, dweight = weight_function(ximag / crad)
        computed_wbs_integrand = (weight * circ / ximag^3) .* dir
        # Tmat = SMatrix{3, 3, T}(0, vtan[3], -vtan[2], -vtan[3], 0, vtan[1], vtan[2], -vtan[1], 0) / vtanmag^2
        # dirtensor = (xi / ximag) * transpose(dir/(vtanmag * ximag))
        # term1 = (weight / ximag^3) .* Tmat
        # term2 = (dweight / (ximag^2 * crad)) .* dirtensor
        # term3 = ((3 * weight) / ximag^3) .* dirtensor
        # volgradtensor = term1 + term2 - term3
    end
    # println("typeof(endofseg) wbs integrand> ", typeof(endofseg))  # DEBUG
    return computed_wbs_integrand, rtnell, endofseg
end