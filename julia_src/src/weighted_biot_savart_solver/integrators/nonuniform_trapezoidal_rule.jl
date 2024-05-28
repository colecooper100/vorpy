using StaticArrays: SVector


###### Integrate along a given path segment ######
# F(x) = \int_a^b f(x) dx \approx \sum_{i=1}^{N} (f(x_{i-1}) + f(x_i))/2 * \Delta x_i
# where \Delta x_i = x_i - x_{i-1}
function nonuniform_trapezoidal_rule_segment(stepsize, fp, vpp1, vpp2, vcr1, vcr2, cir1, cir2)
    # Initialize variables used in the loop
    sol = SVector{3, Float32}(0, 0, 0)
    eval_bs_integrand, ell, endofseg = biot_savart_integrand(Float32(0), fp, vpp1, vpp2, vcr1, vcr2, cir1, cir2)

    # Step through the segment
    while !endofseg
        prev_ell = ell
        prev_eval_bs_integrand = eval_bs_integrand
        eval_bs_integrand, ell, endofseg = biot_savart_integrand(prev_ell + stepsize, fp, vpp1, vpp2, vcr1, vcr2, cir1, cir2)
        
        # Acculmulate the step solutions of
        # the integrand
        sol = sol .+ (prev_eval_bs_integrand .+ eval_bs_integrand) * (ell - prev_ell) / 2
    end
    return sol / (4 * Float32(pi))
end