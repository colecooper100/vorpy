using CUDA
using StaticArrays
using Plots

using weighted_biot_savart_solver
using vortex_paths
using utilities

println("* Imported modules and local scripts")


# GPU kernel
# Make a wrapper to run function on the GPU.
# Kernel functions needs to return `nothing`
function gpu_kernel(vol_rtn,  # return values
                    fp, vpps, crads, circs, stepscalar)  # input values
    
    vol_rtn .= u_wbs_1fp(fp, vpps, crads, circs, stepscalar)

    return nothing
end

println("* GPU kernel defined")

#=================== Test velocity profile ===================#
#=============================================
# Make vortex
A vortex is the collection of the following:
- Path: 3xN matrix where N is the number of
  points in the path. Note: there are either
  N-1 or N segments in the path.
- Core radii: 1xN vector where N is the number
  of points in the path.
- Circulations: 1xN vector where N is the number
  of points in the path.
=============================================#
TYP = Float32  # Probably should be a float
VPPI = SVector{3, TYP}(-10000, 0, 0)
VPPF = SVector{3, TYP}(10000, 0, 0)
NUMSEGS = 1000
CORERADIUS = TYP(5)
CIRCULATION = TYP(10.0)
STEPSCALAR = TYP(0.25)  # Integrator step size = scalar * min(cradI, cradF)

vpps = LineVortex(NUMSEGS, VPPI, VPPF)
crads = ones(TYP, NUMSEGS + 1) .* CORERADIUS
circs = ones(TYP, NUMSEGS + 1) .* CIRCULATION

println("* Vortex defined")

numfps = 10
ZPLANE = TYP(0.0)
fps = zeros(TYP, 3, numfps)
fps[1, :] .= zeros(TYP, numfps)
fps[2, :] .= TYP.(range(1e-3, 60, length=numfps))
fps[3, :] .= ones(TYP, numfps) .* ZPLANE

println("* Field points defined")

#================================================
# Allocate memory on the GPU
Values returned by the GPU need to be passed
by modify elements of mutable objects. This is
because the GPU can't pass values to the CPU
directly, values must be copied from the GPU
to the CPU. This is done by modifying the
elements of a CuArray.
================================================#
cuvpps = CuArray{TYP}(vpps)
cucrads = CuArray{TYP}(crads)
cucircs = CuArray{TYP}(circs)
# curtnvel = CuArray{TYP}(undef, 3)

numrtnvels = zeros(TYP, 3, numfps)
anartnvels = zeros(TYP, 3, numfps)
for i in axes(fps, 2)
  curtnvel = CUDA.zeros(TYP, 3)

  # We don't need to make fp a CuArray because
  # SVectors can be passed to the GPU
  fp = getfp(fps, i)
  println("fp = ", fp)

  anartnvels[:, i] .= u_LineVortex(fp, VPPI, VPPF, crads[1], circs[1])  # Analytical
  # numrtnvels[:, i] .= u_wbs_1fp(fp, vpps, crads, circs, STEPSCALAR)  # CPU
  @device_code_warntype @cuda gpu_kernel(curtnvel, fp, cuvpps, cucrads, cucircs, STEPSCALAR)
  numrtnvels[:, i] .= Array(curtnvel)
end

errvec = anartnvels .- numrtnvels

axislabels = ("x", "y", "z")
markershapes = (:+, :o, :dtriangle)
plt1 = plot(title="Error in velocity profile", xlabel="distance from line", ylabel="Error in velocity")
plt2 = twinx()  # Make a second y-axis
for i in 1:3
  plot!(plt1, fps[2, :], abs.(errvec[i, :]), label=axislabels[i], markershape=markershapes[i], legend=:topleft)
  plot!(plt2, fps[2, :], anartnvels[i, :], label="Ana $(axislabels[i])", ylabel="Velocity Magnitude", markershape=:square, legend=:topright)
  plot!(plt2, fps[2, :], numrtnvels[i, :], label="Num $(axislabels[i])")
end
display(plt1)

 
