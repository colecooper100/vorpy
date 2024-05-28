import numpy as np
from juliacall import Main as jl

# NOTE: vorpy can take up to 20 seconds to load

# Activate Julia environment in the
# current working directory (which should be vorpy)
jl.Pkg.activate('.')

_available_devices = {}

# Load the CUDA Biot-Savart solver if available
try:
    # Load CUDA Biot-Savart function
    jl.include('julia_src/weighted_biot_savart_kernel_cuda.jl')

    # User API
    def _bs_solve_cuda(fps, vpps, vcrds, vcirs):
        return np.transpose(jl.bs_solve_cuda(np.transpose(fps),
                                            np.transpose(vpps), vcrds, vcirs))
    
    _available_devices['cuda'] = _bs_solve_cuda

except:
    print('CUDA version of Biot-Savart solver not available; set bs_solve device to \'cpu\'')

finally:
    # Load CPU Biot-Savart function.
    jl.include('julia_src/weighted_biot_savart_solver_cpu.jl')

    # User API
    def _bs_solve_cpu(fps, vpps, vcrds, vcirs):
        return np.transpose(jl.bs_solve_cpu(np.transpose(fps),
                                            np.transpose(vpps), vcrds, vcirs))
    
    _available_devices['cpu'] = _bs_solve_cpu

    def bs_solve(fps, vpps, vcrds, vcirs, device='cpu'):
        try:
            return _available_devices[device](fps, vpps, vcrds, vcirs)
        except KeyError:
            raise ValueError(f'Invalid device: \'{device}\'. Available devices: {list(_available_devices.keys())}')

