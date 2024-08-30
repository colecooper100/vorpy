# import time
import juliacall
import numpy as np

# Set up Julia
jl = juliacall.newmodule("vorpy")
jl.seval("using Pkg")
jl.Pkg.activate("../../julia")
jl.seval("using weighted_biot_savart_integrator")
jl.seval("using vortex_paths")
# jl.Pkg.status()

###################### OUTPUT FROM JULIACALL ######################
# UserWarning: Julia was started with multiple threads
# but multithreading support is experimental in JuliaCall.
# It is recommended to restart Python with the environment
# variable PYTHON_JULIACALL_HANDLE_SIGNALS=yes set,
# otherwise you may experience segfaults or other crashes.
# Note however that this interferes with Python's own signal
# handling, so for example Ctrl-C will not raise KeyboardInterrupt.
# See https://juliapy.github.io/PythonCall.jl/stable/faq/#Is-PythonCall/JuliaCall-thread-safe?
# for further information. You can suppress this warning
# by setting PYTHON_JULIACALL_HANDLE_SIGNALS=no.
###################### OUTPUT FROM JULIACALL ######################



def wbs_cpu(vrtx, fps, stepscalar, minstepsize, thread):
    """
    wbs_cpu: Weighted Biot-Savart integrator

    # Arguments
    vrtx: vorpath object
    fps: Nx3 array
    stepscalar: float
    minstepsize: float
    threaded: bool

    # Returns
    rtnvals: NxM array, where M is what ever the Julia wbs_cpu
    function returns
    """

    # t0 = time.time_ns()  # TIMING
    rtnvals = jl.wbs_cpu(fps.T,
                         vrtx.vpps.T,
                         vrtx.crads,
                         vrtx.circs,
                         stepsizescalar=stepscalar,
                         minstepsize=minstepsize,
                         threaded=thread)

    # t1 = time.time_ns()  # TIMING

    return rtnvals.to_numpy().T


class vorpath:

    def __init__(self, vpps, crads, circs):
        """
        vpps: Nx3 array
        crads: N array
        circs: N array
        """
        self.vpps = np.array(vpps)
        self.crads = np.array(crads)
        self.circs = np.array(circs)

    def __len__(self):
        return self.vpps.shape[0]
    
    def __repr__(self):
        return f'vorpath(vpps=\n\t{repr(self.vpps)},\n\tcrads=\n\t{repr(self.crads)},\n\tcircs=\n\t{repr(self.circs)})'

    def velfp(self, fps, stepscalar, minstepsize, thread):
        """
        vel: velocity at fps induced by the vortex

        # Arguments
        fps: Nx3 array of floating point numbers
        stepscalar: float
        minstepsize: float
        threaded: bool

        # Returns
        rtnvals: Nx3 array
        """
        return wbs_cpu(self, np.array(fps).reshape(-1, 3), stepscalar, minstepsize, thread)

    def vel(self, stepscalar, minstepsize, thread):
        """
        vel: velocity at each path point

        # Arguments
        stepscalar: float
        minstepsize: float
        threaded: bool

        # Returns
        rtnvals: Nx3 array
        """
        return wbs_cpu(self, self.vpps, stepscalar, minstepsize, thread)
    

    
    

