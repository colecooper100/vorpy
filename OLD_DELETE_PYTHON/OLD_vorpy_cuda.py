import numpy as np
from juliacall import Main as jl

# Activate Julia environment in the
# current working directory (which should be vorpy)
jl.Pkg.activate('.')

# Run the Julia script to load the Biot-Savart
# function.
jl.include('julia_src/weighted_biot_savart_kernel_cuda.jl')  # GPU version
def bs_solve(fps, vpps, vcrds, vcirs):
    return np.transpose(jl.bs_solve_cuda(np.transpose(fps),
                                        np.transpose(vpps), vcrds, vcirs))

