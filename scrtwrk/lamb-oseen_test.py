import time
import juliacall
import numpy as np

# Set up Julia
jl = juliacall.newmodule("lamb_oseen_test")
jl.seval("using Pkg")
jl.Pkg.activate("./julia")
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

def lamb_oseen_test(linelength, numsegments, dtyp, crad=5, circ=1):
    vppF = np.array([linelength/2, 0.0, 0.0], dtype=dtyp)
    vppI = -vppF.copy()
    vpps = np.linspace(vppF, vppI, numsegments+1, dtype=dtyp)
    crads = np.ones(numsegments+1, dtype=dtyp) * dtyp(crad)
    circs = np.ones(numsegments+1, dtype=dtyp) * dtyp(circ)
    NUMFPS = 100
    fps = np.zeros((NUMFPS, 3), dtype=dtyp)
    fps[:, 0] = dtyp(0)
    fps[:, 1] = np.linspace(1, 45, NUMFPS, dtype=dtyp)
    STEPSCALAR = dtyp(1e-6)
    MINSTEPSIZE = dtyp(1e-2)

    t0 = time.time_ns()  # TIMING
    rtnvals = jl.wbs_cpu(fps.T,
                         vpps.T,
                         crads,
                         circs,
                         stepsizescalar=STEPSCALAR,
                         minstepsize=MINSTEPSIZE,
                         threaded=True)

    t1 = time.time_ns()  # TIMING

    anavelsinf = np.zeros((NUMFPS, 3), dtype=dtyp)
    anavelspoly = np.zeros((NUMFPS, 3), dtype=dtyp)
    for i in range(NUMFPS):
        anavelsinf[i] = jl.u_inflong_line(fps[i], vppI, vppF, crads[1], circs[1])
        anavelspoly[i] = jl.u_polyline(fps[i], vpps.T, circs)

    errinf = np.abs(anavelsinf[:, 2] - rtnvals.to_numpy().T[:, 2])
    errpoly = np.abs(anavelspoly[:, 2] - rtnvals.to_numpy().T[:, 2])

    return rtnvals.to_numpy().T, errinf, errpoly, t1 - t0

# Run once (to compile?)
rtn1 = lamb_oseen_test(1, 10, np.float64, crad=5, circ=1)
print('Time (ms):', rtn1[3]/1e6)

# Run the test
LINELENGTHS = [10, 100, 1_000, 10_000]  # [10, 100, 1_000, 10_000]
NUMSEGMENTS = [1, 3, 10, 30, 200, 600, 400, 1200, 2000, 6000]  # [1, 10, 200, 400, 2000]
# Header of table output
print('Line Length, Num Segs, Seg Width, Mean Error, Min Error, Max Error, Time (ms)')  
for i in range(len(LINELENGTHS)):
    for ii in range(len(NUMSEGMENTS)):
        vel, errinf, errpoly, teval = lamb_oseen_test(LINELENGTHS[i], NUMSEGMENTS[ii], np.float64, crad=5, circ=1)
        print(f'{LINELENGTHS[i]}, {NUMSEGMENTS[ii]}, {LINELENGTHS[i]/NUMSEGMENTS[ii]}, {np.mean(errinf)}, {np.min(errinf)}, {np.max(errinf)}, {teval/1e6}')


