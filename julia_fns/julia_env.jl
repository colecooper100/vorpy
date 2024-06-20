###### Paths for commonly accessed directories ######
#====================================================
I found that when I needed to change where a script
was located, I had to change the path in several
files. By using environment variables, I can change
the path in one place and it will be updated in all
the scripts that use it.

IMPORTANT: When using JuliaCall, Julia's working
directory (which can be checked with `pwd()` and
changed with `cd(<path>)`) is determined by the
directory that the Python code is running outof.
To make the paths paths in this script work,
you need to set the global variable JULIA_FNS to 
the path of the julia_fns directory. This can be
done in the Python script that calls the Julia code.
====================================================#
using Pkg: activate, instantiate
activate(JULIA_FNS)  # Activate the Julia project environment
instantiate()  # Install and precompile the project packages (if needed)

UTILITY_FUNCTIONS = string(JULIA_FNS, "/src/utility_functions")
VORTEX_DYNAMICS = string(JULIA_FNS, "/src/vortex_dynamics")
VORTEX_MODELS = string(JULIA_FNS, "/src/vortex_models")
WEIGHTED_BIOT_SAVART_SOLVER_ONE_FIELD_POINT = string(JULIA_FNS, "/src/weighted_biot_savart_solver_one_field_point")
WEIGHTED_BIOT_SAVART_INTEGRATOR = string(WEIGHTED_BIOT_SAVART_SOLVER_ONE_FIELD_POINT, "/weighted_biot_savart_integrator")
WEIGHTED_BIOT_SAVART_INTEGRAND = string(WEIGHTED_BIOT_SAVART_INTEGRATOR, "/weighted_biot_savart_integrand")
WEIGHTED_BIOT_SAVART_INTEGRATOR_METHODS = string(WEIGHTED_BIOT_SAVART_INTEGRATOR, "/weighted_biot_savart_integrator_methods")

println("* Julia environment variables loaded for Vorpy.")