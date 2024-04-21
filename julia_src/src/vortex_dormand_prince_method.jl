# \vec x' = f(t, \vec x)
function _vortex_dormand_prince_step(vpps, vcrds, vcirs, t0, dt)
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

    k1 = _bsfndp(t0, vpps, vcrds, vcirs)
    k2 = _bsfndp(t0 + c2 * dt, vpps .+ dt .* (a21 .* k1), vcrds, vcirs)
    k3 = _bsfndp(t0 + c3 * dt, vpps .+ dt .* (a31 .* k1 .+ a32 .* k2), vcrds, vcirs)
    k4 = _bsfndp(t0 + c4 * dt, vpps .+ dt .* (a41 .* k1 .+ a42 .* k2 .+ a43 .* k3), vcrds, vcirs)
    k5 = _bsfndp(t0 + c5 * dt, vpps .+ dt .* (a51 .* k1 .+ a52 .* k2 .+ a53 .* k3 .+ a54 .* k4), vcrds, vcirs)
    k6 = _bsfndp(t0 + dt, vpps .+ dt .* (a61 .* k1 .+ a62 .* k2 .+ a63 .* k3 .+ a64 .* k4 .+ a65 .* k5), vcrds, vcirs)
    k7 = _bsfndp(t0 + dt, vpps .+ dt .* (a71 .* k1 .+ a73 .* k3 .+ a74 .* k4 .+ a75 .* k5 .+ a76 .* k6), vcrds, vcirs)

    # println("Computing solution...")  # DEBUG
    sol4 = vpps .+ dt .* (b1 .* k1 .+ b3 .* k3 .+ b4 .* k4 .+ b5 .* k5 .+ b6 .* k6)
    sol5 = vpps .+ dt .* (bhat1 .* k1 .+ bhat3 .* k3 .+ bhat4 .* k4 .+ bhat5 .* k5 .+ bhat6 .* k6 .+ bhat7 .* k7)

    # println("Computing local error...")  # DEBUG
    local_err = sqrt(sum((sol5 - sol4) .^ 2) / length(sol4))  # RMS error

    # println("Exiting vortex_dormand_prince_step...")  # DEBUG
    # position, velocity at position and time of solution, local error
    return sol4, _bsfndp(t0 + dt, sol4, vcrds, vcirs), local_err
end

function _new_step_size(dt, local_err, loc_err_tol, global_err_order, safety_factor)
    return Float32(safety_factor * dt * (loc_err_tol / local_err) ^ (1 / (global_err_order + 1)))
end

function vortex_dynamics_adaptive_dormand_prince(vpps_init, vcrds, vcirs, timespan, min_step_size, max_step_size, loc_err_tol; report_interval=Inf32)
    # report_interval: If a floating point number is
    # specified, the function will compute and return
    # the solution of _vdpbsfn(t, u, p) at each time step
    # t_i = t_0 + i * report_step_size where i = 0, 1, 2, ...
    # and p are any parameters other than t and u.

    # Initial dt
    init_vel = _bsfndp(timespan[1], vpps_init, vcrds, vcirs)
    dt = 1f-1 / sqrt(sum(init_vel) .^ 2 / length(init_vel))
    # println("Initial dt: ", dt)  # DEBUG

    time_vec = Array{Float32, 1}([timespan[1]])
    vpps_evol_vec = Array{Float32, 3}([vpps_init;;;])
    vel_vec = Array{Float32, 3}([init_vel;;;])
    local_err_vec = Array{Float32, 1}([0])  # The initial state of system has no error
    report_time = Float32(timespan[1] + report_interval)
    time_step = Float32(timespan[1])
    # println("Enerting solver while loop...")  # DEBUG
    while time_step < timespan[end]
        # println("Current time:" , time_step)  # DEBUG
        if (time_step + dt) > (report_time + report_interval)
            dt = report_time - time_step
        elseif (time_step + dt) > timespan[end]
            dt = timespan[end] - time_step
        elseif dt > max_step_size
            dt = max_step_size
        elseif dt < min_step_size && (time_step + dt != report_time)
            dt = min_step_size
        else
            vpps_step, vel_step, local_err_step = _vortex_dormand_prince_step(vpps_evol_vec[:, :, end], vcrds, vcirs, time_step, dt)
            
            if isnan(local_err_step) || isinf(local_err_step) && ((dt != min_step_size) || (time_step + dt != report_time) || (time_step + dt != timespan[end]))
                println("Time step $(length(time_vec)): The local error is NaN or Inf. Continuing with the smallest possible step size.")
                dt = minimum([min_step_size, report_time - time_step, timespan[end] - time_step])
            else
                # println("Step ", length(time_vec), " accepted! Step size: ", dt)  # DEBUG
                # println("current_time + dt: ", current_time + dt)  # DEBUG
                # println("Report time: ", report_time)  # DEBUG

                if (time_step + dt == report_time) || (time_step + dt == timespan[end])
                    # Update the solution
                    report_time = report_time + report_interval  # next report time
                    time_vec = cat(time_vec, time_step + dt; dims=1)
                    # println("Size vpps_evol_vec: ", size(vpps_evol_vec))  # DEBUG
                    # println("Size sol: ", size(sol))  # DEBUG
                    vpps_evol_vec = cat(vpps_evol_vec, vpps_step; dims=3)
                    vel_vec = cat(vel_vec, vel_step; dims=3)
                    local_err_vec = cat(local_err_vec, local_err_step; dims=1)
                end

                # Update the time
                time_step = time_step + dt

                # Pick a new step size using the local error
                if local_err_step < loc_err_tol
                    dt = _new_step_size(dt, local_err_step, loc_err_tol, 4, 0.9)
                    # println("New step size: ", dt)  # DEBUG
                end
            end
        end
    end

    return vpps_evol_vec, vel_vec, time_vec, local_err_vec
end