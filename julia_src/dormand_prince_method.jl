
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

function constant_v(t, x)
    return 2
end

function time_depend_v(t, x)
    return 2*t
end



vfn(t, x0) = time_depend_v(t, x0)


function dormand_prince_step(stepsize, t, x)
    k1 = vfn(t, x)
    k2 = vfn(t + stepsize/5, x + stepsize/5*k1)
    k3 = vfn(t + 3*stepsize/10, x + 3*stepsize/40*k1 + 9*stepsize/40*k2)
    k4 = vfn(t + 4*stepsize/5, x + 44*stepsize/45*k1 - 56*stepsize/15*k2 + 32*stepsize/9*k3)
    k5 = vfn(t + 8*stepsize/9, x + 19372*stepsize/6561*k1 - 25360*stepsize/2187*k2 + 64448*stepsize/6561*k3 - 212*stepsize/729*k4)
    k6 = vfn(t + stepsize, x + 9017*stepsize/3168*k1 - 355*stepsize/33*k2 + 46732*stepsize/5247*k3 + 49*stepsize/176*k4 - 5103*stepsize/18656*k5)
    k7 = vfn(t + stepsize, x + stepsize*(35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6))

    sol4 = x + stepsize*(5179/57600*k1 + 7571/16695*k3 + 393/640*k4 - 92097/339200*k5 + 187*stepsize/2100*k6 + 1/40*k7)
    sol5 = x + stepsize*(35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6)
    err = abs(sol5 - sol4)
    return sol5, err
end

# function adaptive_stepsize_dormand_prince_method(t, x, tol)
#     stepsize = 1e-3
#     sol5, err = dormand_prince_step(stepsize, t, x)
#     while norm(err) > tol
#         stepsize /= 2
#         sol5, err = dormand_prince_step(stepsize, t, x)
#     end
#     return sol5
# end


################## Test ##################
using Plots

X0 = 3
T0 = 3
TF = 10
# stepsize = 1
# xplt = plot()
# toterrplt = plot()
# locerrplt = plot()
for stepsize in [1e-2, 1e-1, 1, 2, 4]
    tvec = T0:stepsize:TF

    # ana_sol_const_v = X0 .+ 2*(tvec .- T0)  # constant_v
    ana_sol_const_v = X0 - T0^2 .+ tvec.^2  # time_depend_v
    num_sol_const_v = zeros(length(tvec))
    loc_err = zeros(length(tvec))
    num_sol_const_v[1] = X0
    for i in 2:length(tvec)
        num_sol_const_v[i], loc_err[i] = dormand_prince_step(stepsize, tvec[i-1], num_sol_const_v[i-1])
    end

    xplt = plot()
    toterrplt = plot()
    locerrplt = plot()

    plot!(xplt, tvec, ana_sol_const_v, label="Analytical", markershape=:x)
    plot!(xplt, tvec, num_sol_const_v, label="Numerical", markershape=:none)


    plot!(toterrplt, tvec, abs.(ana_sol_const_v - num_sol_const_v), label="Error")
    plot!(locerrplt, tvec, loc_err, label="Local Error")

    display(plot(xplt, toterrplt, locerrplt, layout=(3, 1), legend=true))
end

# display(plot(xplt, toterrplt, locerrplt, layout=(3, 1), legend=true))