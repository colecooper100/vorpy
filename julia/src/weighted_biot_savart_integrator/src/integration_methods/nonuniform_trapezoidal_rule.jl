#============================================
Nonuniform trapezoidal rule
    We assume the values being integrated are
    packed flat in a single SVector. 
============================================#


# F(x) = \int_a^b f(x) dx \approx \sum_{i=1}^{N} (f(x_{i-1}) + f(x_i))/2 * \Delta x_i
# where \Delta x_i = x_i - x_{i-1}
# function nonuniform_trapezoidal_rule(rtnval::AbstractArray{T, 1}, stepsize::T, params::SVector) where {T<:AbstractFloat}  # No return
function nonuniform_trapezoidal_rule(
                                    stepsize::T,
                                    params::SVector) where {T<:AbstractFloat}
    
    # Evaluate integrand function at ell=0
    # to initialize the integrator
    curr_eval, ell, endofseg = integrand(params, T(0))  # ell=0

    # Initialize the return value
    rtnval = curr_eval

    # Step through the segment
    itercount = 1  # DEBUG
    while !endofseg
        itercount += 1  # DEBUG
        # Advance the method by one step
        prev_ell = ell
        prev_eval = curr_eval
        ellmaybe = ell + stepsize
        # Evaluate the integrand at ellmaybe step
        curr_eval, ell, endofseg = integrand(params, ellmaybe)
        # Acculmulate the step solutions of
        # the integrand
        deltaell = ell - prev_ell
        rtnval = rtnval .+ ((prev_eval .+ curr_eval) .* deltaell / T(2))
    end

    return rtnval
end