using StaticArrays: SVector
using LinearAlgebra: norm, cross


# The integrand always expects ell \in [0, seglen]
function biot_savart_integrand(ell, fp, vpp1, vpp2, vcr1, vcr2, cir1, cir2)
    # Get the vortex properties at the given
    # arc length ell
    vpp, vtan, vcr, cir, rtnell, endofseg = vortex_model(ell, vpp1, vpp2, vcr1, vcr2, cir1, cir2)

    # Compute xi: the vector which extends from a
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