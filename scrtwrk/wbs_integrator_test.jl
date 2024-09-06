using Statistics
using StaticArrays
using Plots
using weighted_biot_savart_integrator
using vortex_paths


function test_wbs_integrator(linelen, numsegs, numfps, analyticalsol=false)
    TYP = Float64
    # Test variables
    vppF = TYP[linelen/2, 0, 0]
    vppI = -copy(vppF)
    vpps = linevortex(numsegs, vppI, vppF)
    crads = ones(TYP, numsegs+1) .* TYP(5)
    circs = ones(TYP, numsegs+1)  # Vector{TYP}(1:NUMSEGS)
    fps = zeros(TYP, 3, numfps)
    fps[1, :] .= TYP(0)
    fps[2, :] .= range(1, 45, length=numfps)
    STEPSCALAR = TYP(1e-1)
    MINSTEPSIZE = TYP(1e-3)

    t0 = time_ns()  # TIMING
    #=======================================
    function wbs_cpu(
                fieldpoints,
                vorpathpoints,
                cordradii,
                circulations;
                stepsizescalar::T,
                minstepsize::T,
                threaded::Bool) where {T<:AbstractFloat}
    =======================================#
    rtnvals = wbs_cpu(
                    fps,
                    vpps,
                    crads,
                    circs;
                    stepsizescalar=STEPSCALAR,
                    minstepsize=MINSTEPSIZE,
                    threaded=false)
    
    t1 = time_ns()  # TIMING
    tottimems = (t1-t0) / 1e6  # TIMING
    println("Total eval time: ", tottimems, " ms")  # TIMING
    # println("Numer of field points: ", numfps)  # TIMING
    # timeperfp = tottimems / numfps  # TIMING
    # println("Time per field point: ", timeperfp, " ms")  # TIMING
    # intgsteps = linelen / MINSTEPSIZE  # TIMING
    # timeperintgstep = (timeperfp / intgsteps) * 1e6  # TIMING
    # println("Line length: ", linelen)  # TIMING
    # println("Minimum step size: ", MINSTEPSIZE)  # TIMING
    # println("Integrator steps per line: ", intgsteps)  # TIMING
    # println("Time per integrator step: ", timeperintgstep, " ns")  # TIMING

    if analyticalsol
        anavelsinf = zeros(TYP, 3, numfps)
        for i in 1:numfps
            anavelsinf[:, i] .= u_inflong_line(fps[:, i], vppI, vppF, crads[1], circs[1])
        end
        errvecinf = abs.(anavelsinf[3, :] .- rtnvals[3, :])
        return rtnvals, anavelsinf, errvecinf
    else
        return rtnvals
    end
end



linelengths = [20_000]  # [10, 100, 1000, 10000]
numnumsegs = [20_000] #[ceil(Integer, linelengths[1] / (5 * 7.5))]  # [1, 10, 200, 400, 2000]
println("Number of segments: ", numnumsegs[1])

# test_wbs_integrator(linelen, numsegs, analyticalsol=false)
rtn = test_wbs_integrator(linelengths[1], numnumsegs[1], 100, true)
# println("rtn[1]: ", rtn[1])  # DEBUG

# Plot the error
errscl = 1  # 2.5  # 0.00012 * numnumsegs[1]
println("Error scal: ", errscl)
errmod = abs.((rtn[1][3, :] ./ errscl) - rtn[2][3, :])
println("Mean error: ", mean(errmod))
println("Max error: ", maximum(errmod))
println("Min error: ", minimum(errmod))
plot(errmod, title="Modified Error")

# Plot the velocity profile
plot(rtn[2][3, :], label="Analytical")
plot!(rtn[1][3, :], label="WBS")



# # Header of table output
# print("| Line Length", "\t| Num Segs", "\t| Seg Width", "\t| Mean Error")
# println("\t\t| Min Error", "\t\t| Max Error", "\t\t| Time (ms)", "\t|")
# for i in eachindex(linelengths)
#     for ii in eachindex(numnumsegs)
#         LINELEN = linelengths[i]
#         NUMSEGS = numnumsegs[ii]
#         vppF = TYP[LINELEN/2, 0, 0]
#         vppI = -copy(vppF)
#         vpps = linevortex(NUMSEGS, vppI, vppF)
#         CRADS = ones(TYP, NUMSEGS+1) .* TYP(5)
#         CIRCS = ones(TYP, NUMSEGS+1)  # Vector{TYP}(1:NUMSEGS)
#         NUMFPS = 1000
#         fps = zeros(TYP, 3, NUMFPS)
#         fps[1, :] .= TYP(0)
#         fps[2, :] .= range(1, 45, length=NUMFPS)
#         STEPSCALAR = TYP(1e-6)
#         MINSTEPSIZE = TYP(1e-2)

#     #===================================
#     function wbs_cpu(
#                 fieldpoints::AbstractArray{T, 2},
#                 vorpathpoints::AbstractArray{T, 2},
#                 cordradii::AbstractArray{T, 1},
#                 circulations::AbstractArray{T, 1};
#                 stepsizescalar::T=T(0.25),
#                 minstepsize::T=T(1e-6),
#                 threaded::Bool=false) where {T<:AbstractFloat}
#     ===================================#
#     t0 = time_ns()  # TIMING
#     rtnvals = wbs_cpu(
#                     fps,
#                     vpps,
#                     CRADS,
#                     CIRCS;
#                     stepsizescalar=STEPSCALAR,
#                     minstepsize=MINSTEPSIZE,
#                     threaded=true)
    
#     t1 = time_ns()  # TIMING

#     anavelsinf = zeros(TYP, 3, NUMFPS)
#     anavelspoly = zeros(TYP, 3, NUMFPS)
#     for i in 1:NUMFPS
#         anavelsinf[:, i] .= u_inflong_line(fps[:, i], vppI, vppF, CRADS[1], CIRCS[1])
#         anavelspoly[:, i] .= u_polyline(fps[:, i], vpps, CIRCS)
#     end

#     # println("anavelsinf")
#     # display(anavelsinf)
#     # println("rtnvals")
#     # display(rtnvals)

#     errvecinf = abs.(anavelsinf[3, :] .- rtnvals[3, :])
#     errvecpoly = abs.(anavelspoly[3, :] .- rtnvals[3, :])
#     # println("Max error: ", maximum(errvec))
#     # println("Min error: ", minimum(errvec))
#     # println("Mean error: ", mean(errvec))
#     println("| ", LINELEN, "\t\t| ", NUMSEGS, "\t\t| ", LINELEN/NUMSEGS, "\t\t| ",
#                     mean(errvecinf), "\t| ", minimum(errvecinf), "\t| ", maximum(errvecinf), "\t| ", (t1-t0) / 1e6, "\t|")

#     plt = plot(fps[2, :], anavelsinf[3, :], label="Analytical")
#     plot!(fps[2, :], anavelspoly[3, :], label="Polyline")
#     plot!(fps[2, :], rtnvals[3, :], label="WBS")
#     title!("Z-Velocity vs. Y-Distance From Vortex Line")
#     xlabel!("Y-Distance")
#     ylabel!("Z-Velocity")
#     display(plt)

#     plt = plot(fps[2, :], errvecinf, label="Error (inf)")
#     plot!(fps[2, :], errvecpoly, label="Error (poly)")
#     title!("Error vs. Distance From Vortex Line")
#     xlabel!("Distance from Vortex Line")
#     ylabel!("Absolute Error")
#     display(plt)
#     end
# end
