###### Import modules and local scrips ######
using StaticArrays: SVector
using LinearAlgebra: norm, cross

# Include the weight function
include(string(WEIGHTED_BIOT_SAVART_INTEGRAND, "/bernstein_polynomial_weight.jl"))


###### Set the weight function ######
function weight_function(delta)
    return bernstein_polynomial_weight(delta)
end


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
    vpp, vtan, vcr, cir, rtnell, endofseg = vortex_model(ell, vpp1, vpp2, crad1, crad2, circ1, circ2)

    # Compute xi, the vector that extends from a
    # point on the vortex path to the field point
    xi = fp .- vpp

    if norm(xi) == 0
        return SVector{3, Float32}(0, 0, 0), rtnell, endofseg
    else
        ximag = norm(xi)
        dir = cross(vtan, xi)
        weight = weight_function(ximag / vcr)
        return (weight * cir / ximag^3) .* dir, rtnell, endofseg
    end
end