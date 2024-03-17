import numpy as np
from juliacall import Main as jl

# Activate Julia environment in the
# current working directory (which should be vorpy)
jl.Pkg.activate('.')

_available_devices = {}

try:
    # Run the Julia script to load the CUDA Biot-Savart
    # function.
    jl.include('julia_src/weighted_biot_savart_kernel_cuda.jl')
    # User API
    def _bs_solve_cuda(fps, vpps, vcrds, vcirs):
        return np.transpose(jl.bs_solve_cuda(np.transpose(fps),
                                            np.transpose(vpps), vcrds, vcirs))
    
    _available_devices['cuda'] = _bs_solve_cuda

except:
    print('CUDA version of Biot-Savart solver not available; set bs_solve device to \'cpu\'')

finally:
    # Run the Julia script to load the CPU Biot-Savart
    # function.
    jl.include('julia_src/weighted_biot_savart_kernel_cpu.jl')  # GPU version
    def _bs_solve_cpu(fps, vpps, vcrds, vcirs):
        return np.transpose(jl.bs_solve_cpu(np.transpose(fps),
                                            np.transpose(vpps), vcrds, vcirs))
    
    _available_devices['cpu'] = _bs_solve_cpu

    def bs_solve(fps, vpps, vcrds, vcirs, device='cpu'):
        return _available_devices[device](fps, vpps, vcrds, vcirs)

