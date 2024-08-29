###### Import modules and local scrips ######
# using StaticArrays: SVector



###### Integrate along a given path segment ######
# F(x) = \int_a^b f(x) dx \approx \sum_{i=1}^{N} (f(x_{i-1}) + f(x_i))/2 * \Delta x_i
# where \Delta x_i = x_i - x_{i-1}
function wbs_nonuniform_trapezoidal_rule(stepsize::T,
                                                    fp::SVector{3, T},
                                                    vpp1::SVector{3, T},
                                                    vpp2::SVector{3, T},
                                                    crad1::T,
                                                    crad2::T,
                                                    circ1::T,
                                                    circ2::T) where {T<:AbstractFloat}

    #======================================
    To make this code more modular, I am
    going to assume that we have no idea
    what is in numapproxvals, at most what
    is known is that it is a tuple of values
    which are the computed integrand values
    of the wbs_integrand_function. These
    values will be bindly integrated by
    whatever method is called. Julia is
    smart enough to be able to apply math
    operations to tuples of values. So the
    method looks just like the single
    value case.
    ======================================#
    eval_wbs_integrand, ell, endofseg = wbs_integrand_function(T(0), fp, vpp1, vpp2, crad1, crad2, circ1, circ2)
    numapproxvals = eval_wbs_integrand

    # Step through the segment
    while !endofseg
        prev_ell = ell
        prev_eval_wbs_integrand = eval_wbs_integrand
        eval_wbs_integrand, ell, endofseg = wbs_integrand_function(prev_ell + stepsize, fp, vpp1, vpp2, crad1, crad2, circ1, circ2)
        
        # Acculmulate the step solutions of
        # the integrand
        numapproxvals = numapproxvals .+ (prev_eval_wbs_integrand .+ eval_wbs_integrand) .* ((ell - prev_ell) / T(2))
    end
    return numapproxvals 
end