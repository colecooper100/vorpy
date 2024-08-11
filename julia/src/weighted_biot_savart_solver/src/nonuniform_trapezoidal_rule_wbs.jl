###### Import modules and local scrips ######
# using StaticArrays: SVector

include("weighed_biot_savart_integrand.jl")


###### Integrate along a given path segment ######
# F(x) = \int_a^b f(x) dx \approx \sum_{i=1}^{N} (f(x_{i-1}) + f(x_i))/2 * \Delta x_i
# where \Delta x_i = x_i - x_{i-1}
function nonuniform_trapezoidal_rule_wbs(stepsize::T,
                                                    fp::SVector{3, T},
                                                    vpp1::SVector{3, T},
                                                    vpp2::SVector{3, T},
                                                    crad1::T,
                                                    crad2::T,
                                                    circ1::T,
                                                    circ2::T) where {T<:AbstractFloat}
    # Initialize variables used in the loop
    sol = SVector{3, T}(0, 0, 0)
    computed_wbs_integrand, ell, endofseg = weighted_biot_savart_integrand(T(0), fp, vpp1, vpp2, crad1, crad2, circ1, circ2)
    # println("endofseg nonuniform> ", endofseg)  # DEBUG
    # println("typeof(endofseg) nonuniform> ", typeof(endofseg))  # DEBUG

    # Step through the segment
    while !endofseg
        prev_ell = ell
        prev_eval_bs_integrand = computed_wbs_integrand
        computed_wbs_integrand, ell, endofseg = weighted_biot_savart_integrand(prev_ell + stepsize, fp, vpp1, vpp2, crad1, crad2, circ1, circ2)
        
        # Acculmulate the step solutions of
        # the integrand
        sol = sol .+ (prev_eval_bs_integrand .+ computed_wbs_integrand) * (ell - prev_ell) / 2
    end
    return sol / (4 * T(pi))
end