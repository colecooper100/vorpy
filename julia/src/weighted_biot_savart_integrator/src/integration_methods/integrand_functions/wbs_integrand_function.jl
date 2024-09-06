#==================================================
I originally had the weight function and vortex
interpolator in wbs_integrand_function.jl but I
realized that these files are only used by the
integrand function, so it seemed more natural to
have those files included in the integrand
function. This way the integrand function is more
self-contained and perhaps easier to read.

The methods are still in separate files but now
they can be called specifically if needed rather
than always being included.
==================================================#

# Set the weight function used by the WBS integrand
include("./weight_functions/bernstein_polynomial_weight.jl")
function weight_function(delta::T) where {T<:AbstractFloat}
    return bernstein_polynomial_weight(delta)
end

#======================================================
I like consolidating the two different interpolators
into a single function. This allows me to change
features while leaving the call to the wrapper function
unchanged. Also, I am hoping that after the function is
exited, any intermediate values are garbage collected.
======================================================#
# Set the vortex properties interpolator
# include("./vortex_interpolators/linear_polygonal_path.jl")
# include("./vortex_interpolators/piecewise_linear_vortex.jl")
include("./vortex_interpolators/linear_polygonal_path_both.jl")
function vortex_interpolator(
                        params::SVector{13, T},
                        ell::T) where {T<:AbstractFloat}
    
    # t1 = time_ns()  # TIMING

    # Unpack the parameters
    # seg = SVector{10, T}(vpp1..., vpp2..., crads[indx], crads[indx+1], circs[indx], circs[indx+1])
    # params = SVector{13, T}(fp..., seg...)
    vppI = SVector{3, T}(params[4], params[5], params[6])
    vppF = SVector{3, T}(params[7], params[8], params[9])
    cradI = params[10]
    cradF = params[11]
    circI = params[12]
    circF = params[13]

    # t2 = time_ns()  # TIMING
    vpp, unttanvpp, crad, circ, elltrue, endofseg = linear_polygonal_vortex(vppI, vppF, cradI, cradF, circI, circF, ell)
    # t3 = time_ns()  # TIMING

    # println("Function: ", (t3-t1), " ns")  # TIMING
    # println("Unpack parameters: ", (t2-t1), " ns")  # TIMING
    # println("Interpolate vortex: ", (t3-t2), " ns")  # TIMING
    return vpp, unttanvpp, crad, circ, elltrue, endofseg
end

#===================================================
The reason I am passing the parameters as a single
vector is to make my life easier if what needs to
be passed to the integrand function changes. This
also allows me to make the integrator general and
leave implementation details to the integrand
function.
===================================================#
# The integrand always expects ell \in [0, seglen]
function wbs_integrand_function(
                        params::SVector{13, T},
                        ell::T) where {T<:AbstractFloat}
    
    # t1 = time_ns()  # TIMING7

    # Get the field point from the parameters
    # params = SVector{13, T}(fp..., vpp1..., vpp2..., crads[indx], crads[indx+1], circs[indx], circs[indx+1])
    fp = SVector{3, T}(params[1], params[2], params[3])
    
    # t2 = time_ns()  # TIMING

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
    vpp, unttanvpp, crad, circ, elltrue, endofseg = vortex_interpolator(params, ell)
    
    # t3 = time_ns()  # TIMING
    
    # Compute R, the vector that extends from a
    # point on the vortex path to the field point
    Rvec = fp .- vpp
    # t4 = time_ns()  # TIMING
    Rmag = norm(Rvec)
    # t5 = time_ns()  # TIMING

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
    if Rmag != 0
        # t6 = time_ns()  # TIMING
        # weight = T(1)  # DEBUG
        weight, _ = weight_function(Rmag / crad)
        # t7 = time_ns()  # TIMING
        scl = circ / (4 * T(pi))
        # t8 = time_ns()  # TIMING
        intg = cross(unttanvpp, Rvec) ./ Rmag^3
        # t9 = time_ns()  # TIMING
        rtnvels = scl .* intg .* weight
    else
        # t4 = t5 = t6 = t7 = 0  # TIMING
        # Divide by zero case
        rtnvels = SVector{3, T}(0, 0, 0)
    end

    # t10 = time_ns()  # TIMING

    # println("Integrand function: ", (t10-t1), " ns")  # TIMING
    # println("Interpolate vortex properties: ", (t3-t2), " ns")  # TIMING
    # println("Norm: ", (t5-t4), " ns")  # TIMING
    # println("Weight function: ", (t7-t6), " ns")  # TIMING
    # println("Cross product: ", (t9-t8), " ns")  # TIMING
    return rtnvels, elltrue, endofseg

end  # function wbs_integrand_function

