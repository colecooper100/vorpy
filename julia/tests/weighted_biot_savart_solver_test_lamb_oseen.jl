using CUDA
using Plots
using LinearAlgebra
using weighted_biot_savart_cpu
using weighted_biot_savart_cuda
using vortex_paths
using utilities


t0 = time_ns()

#####################################
# Set solver
# function bsfn(fps, vpps, crads, circs)
#     # Change this to change what solver is used
#     # return u_wbs_cpu(fps, vpps, crads, circs; threaded=true)
#     # return u_wbs_cuda(fps, vpps, crads, circs; numthreads=350)
# end
bsfn = precompile_u_wbs_cuda()

# # Define the GPU kernel function
# function u_wbs_cuda_test(fps::AbstractArray{T, 2},
#     vpps::AbstractArray{T, 2},
#     crads::AbstractArray{T, 1},
#     circs::AbstractArray{T, 1},
#     stepscalar::T=T(0.25)) where {T<:AbstractFloat}

#     curtnvel = CUDA.zeros(T, size(fps)...)

#     # println("* u_wbs_cuda_test *")
#     # println("Field points: ", fps)  # DEBUG
#     # println("Vortex path points: ", vpps)  # DEBUG
#     # println("Core radii: ", crads)  # DEBUG
#     # println("Circulations: ", circs)  # DEBUG
#     # println("Step scalar: ", stepscalar)  # DEBUG

#     # Run the kernel function on the GPU
#     # Make sure to set the number of threads and blocks
#     # The number of blocks tells the GPU how much work
#     # it has to do. If you have one block, and one thread,
#     # then the WBS function is called for the first field.
#     # If you had 10 field points, for the same example,
#     # you would need 10 blocks (or 1 block of 10 threads).
#     numfps = size(fps, 2)
#     numthreads = 1
#     numblocks = ceil(Int, numfps / numthreads)
#     @device_code_warntype @cuda threads=numthreads blocks=numblocks weighted_biot_savart_cuda.u_wbs_1fp_cuda(curtnvel,
#                                                                             CuArray{T}(fps),
#                                                                             CuArray{T}(vpps),
#                                                                             CuArray{T}(crads),
#                                                                             CuArray{T}(circs),
#                                                                             T(stepscalar))

#     return Array(curtnvel)
# end


#========== Compare the velocity profile ==========#
TYP = Float32
VPPI = TYP[-10_000, 0, 0]
VPPF = TYP[10_000, 0, 0]
CORERADIUS = TYP(5)  # was 0.0001
CIRCULATION = TYP(10.0)
ZPLANE = TYP(0.0)
println("* Type: ", TYP)
println("* Initial point: ", VPPI)
println("* Final point: ", VPPF)
println("* Core radius: ", CORERADIUS)
println("* Circulation: ", CIRCULATION)
println("* Z-plane: ", ZPLANE)

println("* Making vortex...")
numsegs = 100  # 100_000
t1 = time_ns()
vpps = LineVortex(numsegs, VPPI, VPPF)
t2 = time_ns()
crads = ones(TYP, numsegs + 1) .* CORERADIUS
t3 = time_ns()
circs = ones(TYP, numsegs + 1) .* CIRCULATION
t4 = time_ns()
println("Done: ", (t4 .- [t1, t2, t3]) ./ 10^9, " s")


# Make the field points
println("* Making field points...")
NUMVELSAMP = numsegs
SAMPLERANGE = (1e-3, 60) # 250)
t1 = time_ns()
fps = zeros(TYP, 3, NUMVELSAMP)
fps[1, :] .= zeros(TYP, NUMVELSAMP)
fps[2, :] .= TYP.(range(SAMPLERANGE..., length=NUMVELSAMP))
fps[3, :] .= ones(TYP, NUMVELSAMP) .* ZPLANE
println("Done: ", (time_ns() - t1) / 10^9, " s")
# println("Size of fps: ", size(fps))  # DEBUG

# Compute analytical solution
println("* Computing velocities analytically... ")
anavels = zeros(TYP, 3, NUMVELSAMP)
t1 = time_ns()
for i in 1:NUMVELSAMP
    fp = getfp(fps, i)
    # println("Field point: ", fp)
    vel = u_LineVortex(fp, VPPI, VPPF, crads[1], circs[1])
    # println("Analytical velocity: ", vel)
    anavels[:, i] .= vel
end
println("Done: ", (time_ns() - t1) / 10^9, " s")
# println("Analytical velocities: ", anavels)  # DEBUG

# Compute numerical solution
println("* Computing velocities numerically... ")
t1 = time_ns()
numvels = bsfn(fps, vpps, crads, circs)
tnumvels = (time_ns() - t1) / 10^9
println("Done: ", tnumvels, " s", " (", tnumvels / numsegs, " s/path-point-pair)")

# Compute and plot the error
println("* Plotting results... ")
# rmserr, errvec = RMSerror(numvels, anavels)
# println("* Total RMS Error: ", rmserr)
pltsamplerange = 1:1:NUMVELSAMP
PLTSINDEPENDENT = fps[2, pltsamplerange]
errvec = abs.(numvels .- anavels)
axeslabel = ["x", "y", "z"]
plt = plot(ylabel="Error", legend=:bottomleft)
plttwinx = twinx()
for i in 1:3
    plot!(plt, PLTSINDEPENDENT, abs.(errvec[i, pltsamplerange]), label="$(axeslabel[i])-error", markershape=:+)
    plot!(plttwinx, PLTSINDEPENDENT, anavels[i, pltsamplerange], label="$(axeslabel[i])-vel(ana)", ylabel="Velocity", legend=:topright, markershape=:x)
    plot!(plttwinx, PLTSINDEPENDENT, numvels[i, pltsamplerange], label="$(axeslabel[i])-vel(num)", markershape=:none)
end
vline!([CORERADIUS, CORERADIUS*5, CORERADIUS*10], label="Core radii (1, 5x, 10x)")
# title!("Error of Velocity Profile\nat z=$(ZPLANE)")
xlabel!("Distance from vortex line")
display(plt)

ttot = time_ns() - t0
println("* Total time: ", ttot * 10^-9, " s")
# (~51s for 6_000 samples and 1000 segments)


# #====================== Timing ======================#
# # numsegs = 1
# # vpps = LineVortex(numsegs, VPPI, VPPF)
# # crads = ones(TYP, numsegs + 1) .* CORERADIUS
# # circs = ones(TYP, numsegs + 1) .* CIRCULATION

# # numfprange = 2:400
# # fprange = (1e-3, 200)
# # tvec = zeros(Float64, 2, length(numfprange))
# # for i in eachindex(numfprange)
# #     # fps = zeros(TYP, 3, numfprange[i])
# #     # fps[2, :] .= TYP.(range(fprange[1], fprange[2], length=numfprange[i]))
# #     # fps[3, :] .= ones(TYP, length(numfprange[i])) .* ZPLANE
# #     fps = 1e-3 .+ 30 .* rand(Float64, (3, numfprange[i]))
# #     println("Field point: ", i)
# #     numsampinavg = 100
# #     ttotthread = 0.0
# #     ttot = 0.0
# #     for j in 1:numsampinavg

# #         # Compute numerical solution
# #         # println("* Computing numerical solution... ")
# #         numvelsthread = bsfn(fps, vpps, crads, circs)
# #         tI = time_ns()
# #         numvelsthread = bsfn(fps, vpps, crads, circs)
# #         ttotthread += (time_ns() - tI) / 10^9

# #         # # Compute numerical solution
# #         # # println("* Computing numerical solution... ")
# #         # numvels = bsfn(fps, vpps, crads, circs)
# #         # tI = time_ns()
# #         # numvels = bsfn(fps, vpps, crads, circs)
# #         # ttot += (time_ns() - tI) / 10^9
        
# #     end
# #     tvec[1, i] = ttotthread / numsampinavg
# #     # tvec[2, i] = ttot / numsampinavg
# # end

# # plot(numfprange, tvec[1, :], label="Threaded")
# # # plot!(numfprange, tvec[2, :], label="Non-threaded")
# # ylims!(0, 0.0005)

# # numsegrange = 1:300
# # tvec = zeros(Float64, 2, numsegrange)
# # for i in eachindex(numsegrange)
# #     vpps = LineVortex(numsegrange[i], VPPI, VPPF)
# #     crads = ones(TYP, numsegrange[i] + 1) .* CORERADIUS
# #     circs = ones(TYP, numsegrange[i] + 1) .* CIRCULATION
# #     println("Number of segments: ", numsegrange[i])
# #     for j in 1:2
# #         tI = time_ns()

# #         # Compute numerical solution
# #         # println("* Computing numerical solution... ")
# #         numvels = bsfn(fps, vpps, crads, circs; threaded=j==1)

# #         tF = time_ns()
# #         tvec[j, i] = (tF - tI) * 10^-9
# #     end
# # end

# # # plot(numsegrange, tvec[1, :], label="First eval")
# # # plot!(numsegrange, tvec[2, :], label="Second eval")
# # plot(tvec[1, :], label="First eval")
# # plot!(tvec[2, :], label="Second eval")
# # ylims!(0, 0.00015)
# # xlims!(0, 350)