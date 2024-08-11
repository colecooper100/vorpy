###### Paths for commonly accessed directories ######
#====================================================
I found that when I needed to change where a script
was located, I had to change the path in several
files. By using environment variables, I can change
the path in one place and it will be updated in all
the scripts that use it.
====================================================#


###### Activate the julia_fns project ######
#==================================================
IMPORTANT: The JULIA_FNS variable needs to be set
to the directory containing julia_env.jl
In the test scripts, the JULIA_FNS variable is set
explicitly.
When using JuliaCall, the global variable JULIA_FNS
can be set with jl.seval('JULIA_FNS = "/path/to/julia_fns"').
in the Python script that calls the Julia code.
==================================================#
using Pkg: activate, instantiate
activate(JULIA_FNS)  # Activate the Julia project environment
instantiate()  # Install and precompile the project packages (if needed)


###### Import modules ######
# I try not to import packages with `using`
# alone because I found reading the code and
# knowing what package a function came from
# was difficult. I also tried using `import`
# but I found that this could create issues
# with IDEs and the code completion. Because
# this was on VSCode, which is the official
# IDE for Julia, I got the impression that
# the community preferred way to import was
# with `using`. I make the functions I am using
# from the package explicit to indicate 
# what came from where. 
using StaticArrays: SVector, SMatrix
using LinearAlgebra: norm, cross


###### Set global variables used by fns ######
UTILITY_FUNCTIONS = string(JULIA_FNS, "/src/utility_functions")
VORTEX_DYNAMICS = string(JULIA_FNS, "/src/vortex_dynamics")
VORTEX_MODELS = string(JULIA_FNS, "/src/vortex_models")
WEIGHTED_BIOT_SAVART_SOLVER_ONE_FIELD_POINT = string(JULIA_FNS, "/src/weighted_biot_savart_solver_one_field_point")
WEIGHTED_BIOT_SAVART_INTEGRATOR = string(WEIGHTED_BIOT_SAVART_SOLVER_ONE_FIELD_POINT, "/weighted_biot_savart_integrator")
WEIGHTED_BIOT_SAVART_INTEGRAND = string(WEIGHTED_BIOT_SAVART_INTEGRATOR, "/weighted_biot_savart_integrand")
WEIGHTED_BIOT_SAVART_INTEGRATOR_METHODS = string(WEIGHTED_BIOT_SAVART_INTEGRATOR, "/weighted_biot_savart_integrator_methods")


###### Include local scripts and set functions ######
# Set the weight function
include(string(WEIGHTED_BIOT_SAVART_INTEGRAND, "/bernstein_polynomial_weight.jl"))
function weight_function(delta)
    return bernstein_polynomial_weight(delta)
end

# Set the model to interpolate properties of the vortex
include(string(VORTEX_MODELS, "/piecewise_linear_vortex_segment_model.jl"))
function vortex_model(ell, vpp1, vpp2, crad1, crad2, circ1, circ2)
    return piecewise_linear_vortex_segment_model(ell, vpp1, vpp2, crad1, crad2, circ1, circ2)
end

# WBS integrand function ######
include(string(WEIGHTED_BIOT_SAVART_INTEGRAND, "/weighed_biot_savart_integrand.jl"))

# Set numerical integration method of WBS integrand
# Bimodal integrator
include(string(WEIGHTED_BIOT_SAVART_INTEGRATOR_METHODS, "/bimodal_integrator_polygonal_segments/bimodal_biot_savart_integrator_polygonal_segments.jl"))
function wbs_integrator(stepsize, fp, vpp1, vpp2, crad1, crad2, circ1, circ2)
    return bimodal_biot_savart_integrator_polygonal_segments(stepsize, fp, vpp1, vpp2, crad1, crad2, circ1, circ2)
end


println("* Julia environment variables loaded for Vorpy.")