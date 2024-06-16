###### Import modules ######
import os
import numpy as np
import juliacall


###### Make new namespace for Julia ######
# In the JuliaCall documents, it is recommended
# to create a new namespace for Julia to avoid
# conflicts with other modules.
# Additionally, when code needs to be evaluated
# in our Julia namespace, we can use the `seval`
# method of the JuliaCall module.
# See https://juliapy.github.io/PythonCall.jl/stable/juliacall/
# for more information.
print('* Creating new Julia namespace...')
jl = juliacall.newmodule('vorpy')


###### Get julia_fns path ######
# The os.path module has a method named `realpath`
# which returns the absolute path of a file or
# directory. We can use this to get the absolute
# path of the `julia_fns` directory.
# See https://docs.python.org/3/library/os.path.html#os.path.realpath
# and https://www.geeksforgeeks.org/python-os-path-realpath-method/#
# The `strict` keyword argument is set to `True`
# so if file or directory does not exist, or is
# not found an exception is raised.
print('* Getting path to julia_fns...')
_julia_fns_path = os.path.realpath("julia_fns", strict=True)
print('* Path to julia_fns:', _julia_fns_path)


###### Get julia_fns path ######
# Make the global variable `JULIA_FNS`, in Julia,
# to store the absolute path to `julia_fns`.
print('* Setting JULIA_FNS variable in Julia...')
jl.seval(f'JULIA_FNS = "{_julia_fns_path}"')


###### Set up our Julia environment variables ######
# The file 'julia_env.jl' sets several
# global variables which are used by the
# vorpy functions written in Julia.
print(f'* Trying to run julia_env.jl in {jl.JULIA_FNS}...')
jl.seval('include(string(JULIA_FNS, "/julia_env.jl"))')


###### Activate the vorpy Julia project ######
# Julia projects are where project dependencies
# are tracked. The files related to this are 
# project.toml and manifest.toml. I have placed
# these in `julia_fns`.
# Activating the project also serves as a check
# to make sure the `JULIA_FNS` variable is set
# correctly.
print('* Activating vorpy Julia project...')
jl.seval('using Pkg')
jl.Pkg.activate(jl.JULIA_FNS)
# # We can make sure we have the correct project activated
# # by checking the status of the project (i.e. print a list
# # of installed packages and their versions).
# jl.Pkg.status()


###### Set up user API to Biot-Savart solvers ######
# Initialize the _WBS_SOLVER_DEVICES dictionary.
# We will determine what devices are available
# to the user below.
_WBS_SOLVER_DEVICES = {}

# Try to load the CUDA Biot-Savart solver. If it works
# add it to the _WBS_SOLVER_DEVICES dictionary.
try:
    print('* Loading CUDA Biot-Savart solver (this may take a few seconds)...')
    
    # Load CUDA Biot-Savart function
    jl.include(jl.JULIA_FNS + '/weighted_biot_savart_solver_cuda.jl')

    def _wbs_solver_cuda(fps, vpps, crads, circs, stepsize):
        return np.transpose(jl.weighted_biot_savart_solver_cuda(np.transpose(fps),
                                                                np.transpose(vpps),
                                                                crads,
                                                                circs,
                                                                stepsizescalar=stepsize))
    
    # Add cuda Biot-Savart solver to the dictionary of
    # available devices. This only happens if there is
    # no error in the try block.
    _WBS_SOLVER_DEVICES['cuda'] = _wbs_solver_cuda

    print('* CUDA Biot-Savart solver loaded.')

except Exception as e:
    print('!! CUDA version of Biot-Savart solver not available; revert to CPU version.')
    print(f'!! Error: {e}')

finally:
    print('* Loading CPU Biot-Savart solver...')

    # Load CPU Biot-Savart function.
    jl.include(jl.JULIA_FNS + '/weighted_biot_savart_solver_cpu.jl')

    def _wbs_solver_cpu(fps, vpps, crads, circs, stepsize):
        return np.transpose(jl.weighted_biot_savart_solver_cpu(np.transpose(fps),
                                                                np.transpose(vpps),
                                                                crads,
                                                                circs,
                                                                stepsizescalar=stepsize))
    
    # Add CPU Biot-Savart solver to the dictionary of
    # available devices.
    _WBS_SOLVER_DEVICES['cpu'] = _wbs_solver_cpu

    print('* CPU Biot-Savart solver loaded.')


    ######## User API ########
    def wbs_solve(fieldpoints,
                  vorpathpoints,
                  corradii,
                  circulations,
                  *,  # Enforce keyword-only arguments
                  device='cpu',
                  datatype=np.float32,
                  stepsizescalar=0.5):
        """
        Solve the weighted Biot-Savart law for a vortical flow at a set of field points.

        ## Parameters
        - fieldpoints: Nx3 array of real values, where N is the number
            of field points.
        - vorpathpoints: Mx3 array of real values, where M is the number
            of points defining the vortex path.
        - corradii: 1D array of real values, where each value is the
            radius of the vortex at the corresponding point in
            `vorpathpoints`.
        - circulations: 1D array of real values, where each value is
            the circulation of the vortex at the corresponding point
            in `vorpathpoints`.
        - device (keyword): string, optional, default='cpu', the device used to
            solve the Biot-Savart law.
        - datatype (keyword): optional, default=np.float32, the
            data type all passed elements are converted to before
            being passed to the Biot-Savart solver. This should
            be a floating-point type.
        - stepsizescalar (keyword): real value, optional, default=0.5, a
            scalar which determines the step sized used by the
            Biot-Savart integrator. The step size is the product
            of this scalar and the minimum user supplied core radius
            of a segment.
        


        ## Future Features
        - Add support for multiple vortices (i.e., return the velocity
            at the given field points due to multiple vortices). This
            should be the sum of the velocities due to each vortex.
        """
        # Check that vorpathpoints, corradii, circulations
        # have the same number of elements. For example,
        # if vorpathpoints has 10 elements (i.e., 10 three
        # vectors), then corradii and circulations should
        # also have 10 elements. 
        if np.shape(corradii)[0] != np.shape(vorpathpoints)[0] or np.shape(circulations)[0] != np.shape(vorpathpoints)[0]:
            raise ValueError(f'corradii has {np.shape(corradii)[0]} elements and circulations has {np.shape(circulations)[0]}, both must have the same number of elements as vorpathpoints, i.e., {np.shape(vorpathpoints)[0]}.')
        
        try:
            print('In wbs_solve', f'using device: {device}')  # DEBUG
            # Convert the user input to numpy arrays of
            # the specified data type.
            return _WBS_SOLVER_DEVICES[device](np.asanyarray(fieldpoints, dtype=datatype),
                                               np.asanyarray(vorpathpoints, dtype=datatype),
                                               np.asanyarray(corradii, dtype=datatype),
                                               np.asanyarray(circulations, dtype=datatype),
                                               datatype(stepsizescalar))
        except KeyError:
            raise ValueError(f'Invalid device: \'{device}\'. Available devices: {list(_WBS_SOLVER_DEVICES.keys())}')

    print(f'!! User API to Biot-Savart solvers set up. Available devices: {list(_WBS_SOLVER_DEVICES.keys())}')


