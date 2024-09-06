

#============================================
Dustin has said to me several times that he
thinks the point separation on the vortex
path should be on the order of a core radius.

When we integrate Biot-Savart, the step size
of the integrator is some fraction of a core
radius. So the spacing of path points can't
smaller than one core radius. But it can be
a multiple of a core radius longer.

Assuming that segments are 1 to a few cores
in length, the simple segmentor will separate
whole segments into analytical and numerical.
The numerical will be used for any segments
inside the cutoff radius (we over use, but
this is because the analytical solution does
not work inside to near the core).
============================================#