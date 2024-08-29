


# The integrand always expects ell \in [0, seglen]
function wbs_integrand_function(
                        params::SVector{13, T},
                        ell::T) where {T<:AbstractFloat}
    
    # Unpack the parameters
    fp = SVector{3, T}(params[1], params[2], params[3])
    vppI = SVector{3, T}(params[4], params[5], params[6])
    vppF = SVector{3, T}(params[7], params[8], params[9])
    cradI = params[10]
    cradF = params[11]
    circI = params[12]
    circF = params[13]

    # Do stuff with parameters...
    # vpp, unttanvpp, elltrue, endofseg = path_interpolator(vppI, vppF, ell)
    # crad, circ = vortex_interpolator(vpp, vppI, vppF, cradI, cradF, circI, circF)
    vpp, unttanvpp, crad, circ, elltrue, endofseg = vortex_interpolator(
                                                            vppI,
                                                            vppF,
                                                            cradI,
                                                            cradF,
                                                            circI,
                                                            circF,
                                                            ell)

    # Compute R, the vector that extends from a
    # point on the vortex path to the field point
    Rvec = fp .- vpp
    Rmag = norm(Rvec)

    # If Rmag == 0 return zero for computed_wbs_integrand
    # and volgradtensor, otherwise do the calculation
    #=================================================
    Note: the return values rtnvelvec and 
    rtnvelgradten are Initialized to zero and then
    later values are assigned to them. This was done
    for 3 reasons:
        1. To make the method GPU compatible
        2. So the variables always had a defined value
        3. To insure that there was only one return 
              statement even though the return values
              are determined by a conditional statement.
              This is mostly to reduce code complexity
              and make easy to update code later.
    =================================================#
    # Initialize return values
    rtnvels = SVector{3, T}(0, 0, 0)
    if Rmag != 0
        weight, _ = weight_function(Rmag / crad)
        rtnvels = (circ * weight / (4 * T(pi) * Rmag^3)) .* cross(unttanvpp, Rvec)
    end

    return rtnvels, elltrue, endofseg

end  # function wbs_integrand_function