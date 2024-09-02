


function linear_polygonal_vortex(
                            vppI::SVector{3, T},
                            vppF::SVector{3, T},
                            cradI::T,
                            cradF::T,
                            circI::T,
                            circF::T,
                            ell::T) where {T<:AbstractFloat}
    
    # Compute segment vector
    segvec = vppF .- vppI
    seglen = norm(segvec)

    # Unit tagent of segment
    rtnunttanseg = segvec ./ seglen

    # Check if at or pass end of segment
    if ell >= seglen
        # At or pass end of segment
        rtnell = seglen
        rtnendofseg = true
        # Set vortex properties to end of segment
        rtnvpp = vppF
        rtncrad = cradF
        rtncirc = circF
    else
        # Not at end of segment
        rtnell = ell
        rtnendofseg = false
        # Interpolate the vortex properties
        scl = rtnell / seglen
        rtnvpp = vppI .+ (scl .* segvec)
        rtncrad = cradI + (scl * (cradF - cradI))
        rtncirc = circI + (scl * (circF - circI))
    end

    return rtnvpp, rtnunttanseg, rtncrad, rtncirc, rtnell, rtnendofseg

end  # function linear_polygonal_path