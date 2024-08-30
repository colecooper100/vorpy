


# The integrand always expects ell \in [0, seglen]
function wbs_integrand_function(
                        params::SVector{13, T},
                        ell::T) where {T<:AbstractFloat}
    
    # Unpack the parameters
    # seg = SVector{10, T}(vpp1..., vpp2..., crads[indx], crads[indx+1], circs[indx], circs[indx+1])
    # params = SVector{13, T}(fp..., seg...)
    fp = SVector{3, T}(params[1], params[2], params[3])
    vppI = SVector{3, T}(params[4], params[5], params[6])
    vppF = SVector{3, T}(params[7], params[8], params[9])
    cradI = params[10]
    cradF = params[11]
    circI = params[12]
    circF = params[13]

    # Do stuff with the parameters...
    #========================================
    vortex_interpolator(
                    vppI::SVector{3, T},
                    vppF::SVector{3, T},
                    cradI::T,
                    cradF::T,
                    circI::T,
                    circF::T,
                    ell::T) where {T<:AbstractFloat}

        return vpp, unttanvpp, crad, circ, elltrue, endofseg
    ========================================#                
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
    Rmag3 = Rmag^3
    if Rmag != 0
        weight, _ = weight_function(Rmag / crad)
        scl = circ / (4 * T(pi))
        intg = cross(unttanvpp, Rvec) ./ Rmag3
        rtnvels = scl .* intg .* weight
    else
        # Divide by zero case
        rtnvels = SVector{3, T}(0, 0, 0)
    end
    # print("vpp: ", vpp)  # DEBUG
    # print(", unttanvpp: ", unttanvpp)  # DEBUG
    # print(", crad: ", crad)  # DEBUG
    # print(", circ: ", circ)  # DEBUG
    # print(", elltrue: ", elltrue)  # DEBUG
    # print(", endofseg: ", endofseg)  # DEBUG
    # print(", Rmag3: ", Rmag3)  # DEBUG
    # print(", rtnvels: ", rtnvels)  # DEBUG
    # println()  # DEBUG
    return rtnvels, elltrue, endofseg

end  # function wbs_integrand_function