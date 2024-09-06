
# using StaticArrays  # Testing function
# using vortex_paths  # Testing function
# using LinearAlgebra  # Testing function


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
    if (ell >= seglen) || (abs(seglen - ell) <= T(0.05) * seglen)
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


# ########################################################
# function function_test(linelen, numsegs, numellsteps, stepscal=1, crad=5, circ=1)
#     TYP = Float64
#     # Test variables
#     vppF = SVector{3, TYP}(linelen/2, 0, 0)
#     vppI = -copy(vppF)
#     vpps = linevortex(numsegs, vppI, vppF)
#     crads = ones(TYP, numsegs+1) .* TYP.(crad)
#     circs = ones(TYP, numsegs+1) .* TYP.(circ)
#     ellstepsize = TYP(stepscal * linelen / numsegs / numellsteps)
#     println("ellstepsize: ", ellstepsize)  # DEBUG
#     println("------START OF TEST------")

#     # Initialize values
#     endofseg = false
#     ellmaybe = TYP(0)
#     for i in 1:numsegs
#         # Get segment
#         vI = SVector{3, TYP}(vpps[:, i]...)
#         vF = SVector{3, TYP}(vpps[:, i+1]...)
#         crI = crads[i]
#         crF = crads[i+1]
#         ciI = circs[i]
#         ciF = circs[i+1]
#         ellstepcounter = 0  
#         while !endofseg
#             ellstepcounter += 1
#             t0 = time_ns()  # TIMING
#             #=======================================
#             function linear_polygonal_vortex(
#                                     vppI::SVector{3, T},
#                                     vppF::SVector{3, T},
#                                     cradI::T,
#                                     cradF::T,
#                                     circI::T,
#                                     circF::T,
#                                     ell::T) where {T<:AbstractFloat}

#             return rtnvpp, rtnunttanseg, rtncrad, rtncirc, rtnell, rtnendofseg
#             =======================================#
#             vpp, unttanseg, crad, circ, ell, endofseg = linear_polygonal_vortex(
#                                                                         vI,
#                                                                         vF,
#                                                                         crI,
#                                                                         crF,
#                                                                         ciI,
#                                                                         ciF,
#                                                                         ellmaybe)

#             t1 = time_ns()  # TIMING
#             println("ellstepcounter: ", ellstepcounter)
#             println("Time to compute one segment: ", (t1-t0)/1e3, " μs")  # TIMING
#             println("endofseg: ", endofseg)  # DEBUG
#             println("ell: ", ell)  # DEBUG
#             ellmaybe += ellstepsize  # make new ellmaybe
#             println("ellmaybe: ", ellmaybe)  # DEBUG
#             println("vpp: ", vpp, ", vppI(seg): ", vpps[:, i], ", vppF(seg): ", vpps[:, i+1])  # DEBUG
#         end
#     end

#     return nothing
# end

# linelengths = [2]  # [10, 100, 1000, 10000]
# numnumsegs = [1]  # [1, 10, 200, 400, 2000]
# # function function_test(linelen, numsegs, numellsteps, crad=5, circ=1)
# function_test(2, 1, 5, 1.19)


