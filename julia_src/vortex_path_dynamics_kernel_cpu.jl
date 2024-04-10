include("weighted_biot_savart_kernel_cpu.jl")
# include("src/dormand_prince_method.jl")



#===========================================
Goal: Implement a method to evolve the
vortex path using the Dormand-Prince method.
===========================================#

# The function to solve
odefn(t, fps, vcrds, vcirs) = bs_solve_cpu(fps, fps, vcrds, vcirs)


# \vec x' = f(t, \vec x)
function vortex_dormand_prince_step(vpps, vcrds, vcirs, t0, stepsize)
    # println("Entering vortex_dormand_prince_step...")  # DEBUG
    
    c2 = Float32(1/5)
    a21 = Float32(1/5)
    c3 = Float32(3/10)
    a31 = Float32(3/40)
    a32 = Float32(9/40)
    c4 = Float32(4/5)
    a41 = Float32(44/45)
    a42 = Float32(-56/15)
    a43 = Float32(32/9)
    c5 = Float32(8/9)
    a51 = Float32(19372/6561)
    a52 = Float32(-25360/2187)
    a53 = Float32(64448/6561)
    a54 = Float32(-212/729)
    a61 = Float32(9017/3168)
    a62 = Float32(-355/33)
    a63 = Float32(46732/5247)
    a64 = Float32(49/176)
    a65 = Float32(-5103/18656)
    a71 = Float32(35/84)
    a73 = Float32(500/1113)
    a74 = Float32(125/192)
    a75 = Float32(-2187/6784)
    a76 = Float32(11/84)
    b1 = Float32(35/384)
    b3 = Float32(500/1113)
    b4 = Float32(125/192)
    b5 = Float32(-2187/6784)
    b6 = Float32(11/84)
    bhat1 = Float32(5179/57600)
    bhat3 = Float32(7571/16695)
    bhat4 = Float32(393/640)
    bhat5 = Float32(-92097/339200)
    bhat6 = Float32(187/2100)
    bhat7 = Float32(1/40)

    # println("Computing ODE solver stages...")  # DEBUG
    # println("\t k1...")  # DEBUG
    k1 = odefn(t0, vpps, vcrds, vcirs)
    # println("\t k2...")  # DEBUG
    k2 = odefn(t0 + c2 * stepsize, vpps .+ stepsize .* (a21 .* k1), vcrds, vcirs)
    # println("\t k3...")  # DEBUG
    k3 = odefn(t0 + c3 * stepsize, vpps .+ stepsize .* (a31 .* k1 .+ a32 .* k2), vcrds, vcirs)
    # println("\t k4...")  # DEBUG
    k4 = odefn(t0 + c4 * stepsize, vpps .+ stepsize .* (a41 .* k1 .+ a42 .* k2 .+ a43 .* k3), vcrds, vcirs)
    # println("\t k5...")  # DEBUG
    k5 = odefn(t0 + c5 * stepsize, vpps .+ stepsize .* (a51 .* k1 .+ a52 .* k2 .+ a53 .* k3 .+ a54 .* k4), vcrds, vcirs)
    # println("\t k6...")  # DEBUG
    k6 = odefn(t0 + stepsize, vpps .+ stepsize .* (a61 .* k1 .+ a62 .* k2 .+ a63 .* k3 .+ a64 .* k4 .+ a65 .* k5), vcrds, vcirs)
    # println("\t k7...")  # DEBUG
    k7 = odefn(t0 + stepsize, vpps .+ stepsize .* (a71 .* k1 .+ a73 .* k3 .+ a74 .* k4 .+ a75 .* k5 .+ a76 .* k6), vcrds, vcirs)

    # println("Computing solution...")  # DEBUG
    sol4 = vpps .+ stepsize .* (b1 .* k1 .+ b3 .* k3 .+ b4 .* k4 .+ b5 .* k5 .+ b6 .* k6)
    sol5 = vpps .+ stepsize .* (bhat1 .* k1 .+ bhat3 .* k3 .+ bhat4 .* k4 .+ bhat5 .* k5 .+ bhat6 .* k6 .+ bhat7 .* k7)

    # println("Computing local error...")  # DEBUG
    local_err = sqrt(sum((sol5 - sol4) .^ 2) / length(sol4))  # RMS error

    # println("Exiting vortex_dormand_prince_step...")  # DEBUG
    return sol4, local_err
end

function vortex_path_dynamics_adaptive_dormand_prince(vpps, vcrds, vcirs, timespan, min_step_size, max_step_size, loc_err_tol; report_step_size=Inf32)
    # report_step_size: If not nothing, the function will
    # compute and return the solution at each time step
    # t_i = t_0 + i * report_step_size

    

    # Initial stepsize
    initial_val_fn = odefn(timespan[1], vpps, vcrds, vcirs)
    stepsize = 1f-1 / sqrt(sum(initial_val_fn) .^ 2 / length(initial_val_fn))
    # println("Initial stepsize: ", stepsize)  # DEBUG

    time_vec = [timespan[1]]
    vpps_evol_vec = [vpps]
    step_err_vec = [0f0]  # The initial state of system has no error
    report_time = timespan[1] + report_step_size
    current_time = timespan[1]
    # println("Enerting solver while loop...")  # DEBUG
    while current_time < timespan[end]
        if (current_time + stepsize) > (report_time + report_step_size)
            stepsize = report_time - current_time
        elseif (current_time + stepsize) > timespan[end]
            stepsize = timespan[end] - current_time
        elseif stepsize > max_step_size
            stepsize = max_step_size
        elseif stepsize < min_step_size && (current_time + stepsize != report_time)
            stepsize = min_step_size
        else
            sol, local_err = vortex_dormand_prince_step(vpps_evol_vec[end], vcrds, vcirs, current_time, stepsize)
            
            if isnan(local_err) || isinf(local_err) && ((stepsize != min_step_size) || (current_time + stepsize != report_time) || (current_time + stepsize != timespan[end]))
                println("Time step $(length(time_vec)): The local error is NaN or Inf. Continuing with the smallest possible step size.")
                stepsize = minimum([min_step_size, report_time - current_time, timespan[end] - current_time])
            else
                # println("Step ", length(time_vec), " accepted! Step size: ", stepsize)  # DEBUG
                # println("current_time + stepsize: ", current_time + stepsize)  # DEBUG
                # println("Report time: ", report_time)  # DEBUG

                if (current_time + stepsize == report_time) || (current_time + stepsize == timespan[end])
                    # Update the solution
                    report_time = report_time + report_step_size  # next report time
                    push!(time_vec, current_time + stepsize)
                    push!(vpps_evol_vec, sol)
                    push!(step_err_vec, local_err)
                end

                # Update the time
                current_time = current_time + stepsize

                # Pick a new step size using the local error
                if local_err < loc_err_tol
                    stepsize = new_step_size(stepsize, local_err, loc_err_tol, 4, 0.9)
                    # println("New step size: ", stepsize)  # DEBUG
                end
            end
        end
    end

    return vpps_evol_vec, time_vec, step_err_vec
end

function new_step_size(stepsize, local_err, loc_err_tol, global_err_order, safety_factor)
    return safety_factor * stepsize * (loc_err_tol / local_err) ^ (1 / (global_err_order + 1))
end


# ################### DEBUGGING ###################
# using Plots
# using BenchmarkTools

# #================================================
# Generate a vortex ring (points, core radii, and
# circulation).
# ================================================#
# # Vortex paramters
# NUMSEGS = 10
# RINGCENTER = (0, 0, 0)
# RINGRADIUS = 5
# CORERADIUS = 0.5
# TSPAN = (0f0, 1f2)
# println("Number of segments: ", NUMSEGS)
# println("Ring center: ", RINGCENTER)
# println("Ring radius: ", RINGRADIUS)
# println("Core radius: ", CORERADIUS)
# println("Time span: ", TSPAN)

# # Generate the vortex points
# theta = range(0, 2 * pi, length=NUMSEGS+1)
# vpx = RINGRADIUS .* cos.(theta) .+ RINGCENTER[1]
# vpy = RINGRADIUS .* sin.(theta) .+ RINGCENTER[2]
# vpz = zeros(size(vpx)...) .+ RINGCENTER[3]
# vpps = stack([vpx, vpy, vpz], dims=1)
# # To prevent numerical errors, explicitly set
# # the last point to the the first.
# vpps[:, end] = vpps[:, 1]

# # display(plot(vpps[1, :], vpps[2, :], vpps[3, :], label="Vortex points"))

# # Generate the core radii
# vcrds = ones(NUMSEGS+1) * CORERADIUS
# vcrds[end] = vcrds[1]

# # Generate the circulation
# vcirs = ones(NUMSEGS+1) * 1.0
# vcirs[end] = vcirs[1]

# # rtnvels = bs_solve_cpu(vpps, vpps, vcrds, vcirs)

# SOL, TSTEPS, ERR = vortex_path_dynamics_adaptive_dormand_prince(vpps, vcrds, vcirs, TSPAN, 1f-2, Inf32, 1f-1; report_step_size=1.7f1)
# evolplt = plot()
# for i in eachindex(SOL)
#     plot!(evolplt, SOL[i][1, :], SOL[i][2, :], SOL[i][3, :], label=TSTEPS[i])
#     # plt = plot(SOL[i][1, :], SOL[i][2, :], SOL[i][3, :], label=TSTEPS[i], marker=:auto)
#     # zlims!(-.1, 0.75)
#     # display(plt)
# end
# display(evolplt)