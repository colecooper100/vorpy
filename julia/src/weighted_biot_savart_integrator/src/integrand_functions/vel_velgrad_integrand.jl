#==================================================
I was originally going to separate the integrands
for the velocity and velocity gradient. I had just
finished making my first iteration of the joint
integrand method and that performed poorly. I was
trying to separate things to see if that reduce
the complexity of the functions being run.

However, I realized that weight_function returns
the weight and its derivative (with the derivative
being very cheap to compute). Dustin has said that
the weight function is the most expensive part of
the integrand. So, it would be better to do one
evaluation of the weight function and get its
derivative for free rather than two evaluations
(once for the computing the velocity and the other
for the velocity gradient). I considered passing
the derivative of the weight function back and
then passing that to the velocity gradient function
but at that point why not just compute them at the
same time in the same integrand.

I'm going to try to do the joint integrand but
make the out put being integrated a flat SVector.
Maybe the GPU doesn't do math over a tuple as
efficiently as it would an SVector.
==================================================#
function vel_velgrad_integrand(
            #============================================
            I am trying to keep the integrator general
            so all that I need to change is the integrand.
            The only thing assumed is that the integral
            is over a line in some 3D volume. Thus we
            have the input of the arc length variable
            ell and everything else is considered
            parameters to the integrand. It is left to
            the integrand function to unpack the
            parameters vector and do the computation.

            The integrator will not know what parameters
            it is getting but the integrand function will,
            so the params and the output of the integrand
            function should be typed!
            
            The return of the integrand function are the
            values being integrated (i.e., the evaluation
            of the integrand at that arc length ell).

            params = SVector{13, T}(FP..., vpprops...)
            Note: vpprops here are the properties
            defining the vortex which are used by the
            path_interpolator and vortex_interpolator
            ============================================#
            params::SVector{13, T},
            ell::T)::Tuple{SVector{12, T}, T, Bool} where {T<:AbstractFloat}
    
    # Unpack the parameters
    fp = @inbounds SVector{3, T}(params[1], params[2], params[3])
    vppI = @inbounds SVector{3, T}(params[4], params[5], params[6])
    vppF = @inbounds SVector{3, T}(params[7], params[8], params[9])
    cradI = @inbounds params[10]
    cradF = @inbounds params[11]
    circI = @inbounds params[12]
    circF = @inbounds params[13]

    vpp, unttanvpp, elltrue, endofseg = path_interpolator(vppI, vppF, ell)
    crad, circ = vortex_interpolator(vpp, vppI, vppF, cradI, cradF, circI, circF)

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
    rtnvelvec = SVector{3, T}(0, 0, 0)
    rtnvelgradten = SMatrix{3, 3, T}(0, 0, 0, 0, 0, 0, 0, 0, 0)
    if Rmag != 0
        Rmag3 = Rmag^3
        weight, dweight = weight_function(Rmag / crad)

        # Velocity vector
        rtnvelvec = (weight / Rmag3) .* cross(unttanvpp, Rvec)

        # Velocity gradient tensor
        Imat = SMatrix{3, 3, T}(1, 0, 0, 0, 1, 0, 0, 0, 1)
        trm1 = ((dweight / Rmag3) - (3 * weight / (Rmag3 * Rmag))) .* (Rvec ./ Rmag)
        trm2 = (weight / Rmag3) .* cross(Imat, unttanvpp)
        trm3 = transpose(cross(unttanvpp, Rvec))
        rtnvelgradten = ((trm1 * trm3) .- trm2)
    end
    
    # Flatten and pack the values being integrated
    rtnvec = (circ / (4 * T(pi))) .* SVector{12, T}(rtnvelvec..., rtnvelgradten...)
    return rtnvec, elltrue, endofseg

end  # function vel_velgrad_integrand

