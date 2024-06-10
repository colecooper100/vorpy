###### Paths for commonly accessed directories ######
#====================================================
I found that when I needed to change where a script
was located, that I had to change the path in several
places. By using environment variables, I can change
the path in one place and it will be updated in all
the scripts that use it.
===================================================#
# pwd() returns the "present working directory". For
# this project, pwd() should return the path to the
# vorpy directory.
VORPY = pwd()
UTILITY_FUNCTIONS = string(VORPY, "/julia_src/src/utility_functions")
VORTEX_DYNAMICS = string(VORPY, "/julia_src/src/vortex_dynamics")
VORTEX_MODELS = string(VORPY, "/julia_src/src/vortex_models")
WEIGHTED_BIOT_SAVART_SOLVER_ONE_FIELD_POINT = string(VORPY, "/julia_src/src/weighted_biot_savart_solver_one_field_point")
WEIGHTED_BIOT_SAVART_INTEGRATOR = string(WEIGHTED_BIOT_SAVART_SOLVER_ONE_FIELD_POINT, "/weighted_biot_savart_integrator")
WEIGHTED_BIOT_SAVART_INTEGRAND = string(WEIGHTED_BIOT_SAVART_INTEGRATOR, "/weighted_biot_savart_integrand")
WEIGHTED_BIOT_SAVART_INTEGRATOR_METHODS = string(WEIGHTED_BIOT_SAVART_INTEGRATOR, "/weighted_biot_savart_integrator_methods")