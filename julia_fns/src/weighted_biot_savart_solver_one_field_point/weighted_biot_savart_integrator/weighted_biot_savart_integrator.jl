#============================================
This script includes all of the functions
required for the BS integrator. It is kind of
like a preference file where various functions
can be swapped in and out.
============================================#


###### Set the model of the vortex ######
include(string(VORTEX_MODELS, "/piecewise_linear_vortex_segment_model.jl"))
function vortex_model(ell, vpp1, vpp2, crad1, crad2, circ1, circ2)
    return piecewise_linear_vortex_segment_model(ell, vpp1, vpp2, crad1, crad2, circ1, circ2)
end


###### BS integrand function ######
include(string(WEIGHTED_BIOT_SAVART_INTEGRAND, "/weighed_biot_savart_integrand.jl"))


###### Set numerical integration method ######
# # Trapezoidal rule
# include(string(WEIGHTED_BIOT_SAVART_INTEGRATOR_METHODS, "/nonuniform_trapezoidal_rule/biot_savart_nonuniform_trapezoidal_rule.jl"))
# function wbs_integrator(stepsize, fp, vpp1, vpp2, crad1, crad2, circ1, circ2)
#     return biot_savart_nonuniform_trapezoidal_rule(stepsize, fp, vpp1, vpp2, crad1, crad2, circ1, circ2)
# end

# Bimodal integrator
include(string(WEIGHTED_BIOT_SAVART_INTEGRATOR_METHODS, "/bimodal_integrator_polygonal_segments/bimodal_biot_savart_integrator_polygonal_segments.jl"))
function wbs_integrator(stepsize, fp, vpp1, vpp2, crad1, crad2, circ1, circ2)
    return bimodal_biot_savart_integrator_polygonal_segments(stepsize, fp, vpp1, vpp2, crad1, crad2, circ1, circ2)
end




