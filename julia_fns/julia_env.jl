###### Paths for commonly accessed directories ######
#====================================================
I found that when I needed to change where a script
was located, I had to change the path in several
files. By using environment variables, I can change
the path in one place and it will be updated in all
the scripts that use it.
===================================================#
# pwd() returns the "present working directory". For
# this project, pwd() should return the path to the
# vorpy directory.
# VORPY = pwd()
# JULIA_FNS = string(VORPY, "/julia_fns")
#================================================
IMPORTANT: When using JuliaCall, Julia's working
directory (which can be checked with `pwd()` and
changed with `cd(<path>)`) is determined by the
directory that the Python code is running outof.
To make these paths work, I could hardcode the full
path to the `julia_fns` directory, but I want
code to be portable. So, when a Julia function
is called from Python, I will first set
`JULIA_FNS` (in Julia --through Python) to the
the path for `julia_fns`. The rest should
work as expected.
================================================#
UTILITY_FUNCTIONS = string(JULIA_FNS, "/src/utility_functions")
VORTEX_DYNAMICS = string(JULIA_FNS, "/src/vortex_dynamics")
VORTEX_MODELS = string(JULIA_FNS, "/src/vortex_models")
WEIGHTED_BIOT_SAVART_SOLVER_ONE_FIELD_POINT = string(JULIA_FNS, "/src/weighted_biot_savart_solver_one_field_point")
WEIGHTED_BIOT_SAVART_INTEGRATOR = string(WEIGHTED_BIOT_SAVART_SOLVER_ONE_FIELD_POINT, "/weighted_biot_savart_integrator")
WEIGHTED_BIOT_SAVART_INTEGRAND = string(WEIGHTED_BIOT_SAVART_INTEGRATOR, "/weighted_biot_savart_integrand")
WEIGHTED_BIOT_SAVART_INTEGRATOR_METHODS = string(WEIGHTED_BIOT_SAVART_INTEGRATOR, "/weighted_biot_savart_integrator_methods")

println("* Julia environment variables loaded for Vorpy.")