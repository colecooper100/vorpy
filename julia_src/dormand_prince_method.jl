#==========================================
Use the Dormand-Prince method to
solve:
    \dot x(t) = v(t)
for:
    - [ ] constant v
    - [ ] v(t) = x0 + 9.8*t
        (assume x0 = 0) answer should be
        x(t) = x0 + v*t + 0.5*9.8*t^2

Once RKDP method is validated,
implement for the vortex path.

Start with a constant stepsize then 
make an adaptive method
==========================================#


# \vec x' = f(t, \vec x)
function dormand_prince_step(fn, t0, x0, stepsize)
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
    c6 = Float32(1)
    a61 = Float32(9017/3168)
    a62 = Float32(-355/33)
    a63 = Float32(46732/5247)
    a64 = Float32(49/176)
    a65 = Float32(-5103/18656)
    c7 = Float32(1)
    a71 = Float32(35/84)
    a72 = Float32(0)
    a73 = Float32(500/1113)
    a74 = Float32(125/192)
    a75 = Float32(-2187/6784)
    a76 = Float32(11/84)
    b1 = Float32(35/384)
    b2 = Float32(0)
    b3 = Float32(500/1113)
    b4 = Float32(125/192)
    b5 = Float32(-2187/6784)
    b6 = Float32(11/84)
    bhat1 = Float32(5179/57600)
    bhat2 = Float32(0)
    bhat3 = Float32(7571/16695)
    bhat4 = Float32(393/640)
    bhat5 = Float32(-92097/339200)
    bhat6 = Float32(187/2100)
    bhat7 = Float32(1/40)

    k1 = fn(t0, x0)
    k2 = fn(t0 + c2 * stepsize, x0 .+ stepsize * (a21 .* k1))
    k3 = fn(t0 + c3 * stepsize, x0 .+ stepsize * (a31 .* k1 .+ a32 .* k2))
    k4 = fn(t0 + c4 * stepsize, x0 .+ stepsize * (a41 .* k1 .+ a42 .* k2 .+ a43 .* k3))
    k5 = fn(t0 + c5 * stepsize, x0 .+ stepsize * (a51 .* k1 .+ a52 .* k2 .+ a53 .* k3 .+ a54 .* k4))
    k6 = fn(t0 + c6 * stepsize, x0 .+ stepsize * (a61 .* k1 .+ a62 .* k2 .+ a63 .* k3 .+ a64 .* k4 .+ a65 .* k5))
    k7 = fn(t0 + c7 * stepsize, x0 .+ stepsize * (a71 .* k1 .+ a72 .* k2 .+ a73 .* k3 .+ a74 .* k4 .+ a75 .* k5 .+ a76 .* k6))

    sol4 = x0 .+ stepsize * (b1 .* k1 .+ b2 .* k2 .+ b3 .* k3 .+ b4 .* k4 .+ b5 .* k5 .+ b6 .* k6)
    sol5 = x0 .+ stepsize * (bhat1 .* k1 .+ bhat2 .* k2 .+ bhat3 .* k3 .+ bhat4 .* k4 .+ bhat5 .* k5 .+ bhat6 .* k6 .+ bhat7 .* k7)

    local_err = sqrt(sum((sol5 - sol4) .^ 2) / length(sol4))  # RMS error

    return sol4, local_err
end

function dormand_prince_adaptive_step(fn, tspan, x0, min_step_size, max_step_size, loc_err_tol)
    # Initial stepsize
    # initial_val_fn = fn(tspan[1], x0)
    # stepsize = 1f-1 / sqrt(sum(initial_val_fn) .^ 2 / length(initial_val_fn))
    stepsize = 1f0  # DEBUG
    # print("Initial stepsize: ", stepsize)  # DEBUG

    tsteps = [tspan[1]]
    x = [x0]
    steperr = []
    while tsteps[end] <= tspan[2]
        sol, local_err = dormand_prince_step(fn, tsteps[end], x[end], stepsize)

        #===========================================
        Here, we just decrease the stepsize by a
        factor of 2 and use a simple method for
        guessing the \Delta t_{n+1} step size.
        However, more sophisticated method for
        selecting a stepsize exist. This will be
        implemented later.
        ===========================================#
        if local_err > loc_err_tol && min_step_size <= stepsize <= max_step_size
            # println("The local error is greater than the specified error tolerance of ", loc_err_tol, ". Decreasing the step size by a factor of 2.")  # DEBUG
            stepsize /= 2
        elseif stepsize > max_step_size 
            println("The step size has been set to ", stepsize, " which is larger than the specified maximum step size of ", max_step_size, ". Try increasing the maximum step size. Continuing with the maximum step size.")
            stepsize = max_step_size
        elseif local_err > loc_err_tol && stepsize < min_step_size
            println("The step size has been set to ", stepsize, " but the local error is ", local_err, " which is larger than the specified error tolerance of ", loc_err_tol, ". Continuing with the minimum step size.")
            stepsize = min_step_size
        else
            push!(tsteps, tsteps[end] + stepsize)
            push!(x, sol)
            push!(steperr, local_err)
            # Set the stepsize for the next step
            ssize = comp_step_size(stepsize, local_err, loc_err_tol, 4, 0.9)  # DEBUG
            println("Step size: ", stepsize, " Local error: ", local_err, " Next stepsize: ", ssize)  # DEBUG
            # stepsize = 1f0  # DEBUG
            stepsize = ssize
            # stepsize = 1f-1 / sqrt(sum(fn(tsteps[end], sol) .^ 2) / length(initial_val_fn))
        end
    end

    return x, tsteps, steperr
    # return stack(x), tsteps, steperr
end

function comp_step_size(stepsize, local_err, loc_err_tol, global_err_order, safety_factor)
    return safety_factor * ((loc_err_tol / local_err) * stepsize ^ (global_err_order + 1)) ^ (1 / (global_err_order + 1))
end


################## Test ##################
using Plots

# Problem: x' = 2
# Solution: x(t) = 2t
function scalar_time_independent(t, x)
    return 2
end

function scalar_time_independent_sol(t, x)
    return 2t
end

# Problem: x' = 2t
# Solution: x(t) = t^2
function scalar_time_depend(t, x)
    return 2*t
end

function scalar_time_depend_sol(t, x)
    return t^2
end

# Solution: x(t) = [sin(t), cos(t)]
function vector_time_independent(t, x)
    return [x[2], -x[1]]
end

function vector_time_independent_sol(t, x)
    return [sin(t), cos(t)]
end


func = vector_time_independent
solfn = vector_time_independent_sol
t0 = Float32(0)
tf = Float32(10)
tspan = [t0, tf]
x0 = [0f0, 1f0]
min_step_size = 1f-6
max_step_size = 5f0

println("Initial time: ", t0)
println("Final time: ", tf)
println("Initial condition: ", x0)
println("Minimum step size: ", min_step_size)
println("Maximum step size: ", max_step_size)

xvec, timesteps, steperror = dormand_prince_adaptive_step(func, tspan, x0, min_step_size, max_step_size, 1f-3)
xvec = stack(xvec)

xvecana = [solfn(timesteps[1], x0)]
for i in 2:length(timesteps)
    append!(xvecana, [solfn(timesteps[i], xvecana[end])])
end
xvecana = stack(xvecana)

plot(timesteps, [xvecana[1, :], xvecana[2, :]], label="Analytical", markershape=:x)
plot!(timesteps, [xvec[1, :], xvec[2, :]], label="Numerical", markershape=:none)
# # plot!(timesteps, [xvec2[1, :], xvec2[2, :]], label="Numerical2", markershape=:none)