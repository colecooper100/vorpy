using LinearAlgebra
using Integrals


################ Helper functions ################

function sciml_integrals(fn, a, b)
    #=
    sciml_integrals(fn, a, b) = \int_a^b fn(u, p) du: the
        integrand fn can be a scalar or vector valued function.
        u is the independent variable and p (optional) is one
        or more parameters.
    =#
    prob = IntegralProblem(fn, a, b)
    sol = solve(prob, HCubatureJL())
    return sol.u
end

function uniform_trapezoidal_rule(fn, a, b; numsteps=100)
    u = range(a, b, length=numsteps)
    return (0.5 .* (fn(a) + fn(b)) + sum(fn.(u[2:end-1]))) .* (b - a) ./ numsteps
end

# # The start of making a unit test for sciml_integrals
# println("Error (sciml_integrals): ", sciml_integrals((x, p)->sin.(x), 0, pi) - (-cos(pi) + cos(0)))


function weighted_biot_savart_kernel(
    fieldpoint,
    vortexpath,
    vortexcore,
    weightfn,
    integrator;
    circulation=1)
    #=
    vortexpath: 3 x N array of points defining the vortex path.
        The vortex path is assumed to be a piecewise linear
        curve.

    vortexcore: N array defining the radius of the vortex core
        at each point along the vortex path. The vortex core
        is assumed to smoothly vary along the vortex path.

    weightfn(\Delta) = float: \Delta is some measure of the
        distance from the vortex path to the field point.

    integrator(fn, a, b) = \int_a^b fn(\ell) d\ell

    circulation: N - 1 array defining vortex strength for each
        vortex path segment
    =#
    # Step through each segment of the vortex path
    rnt_vel = zeros(3)
    for i in 1:(size(vortexpath, 2) -1)
        vseg = vortexpath[:, i+1] - vortexpath[:, i]
        vsegmag = norm(vseg)
        vsegtan = vseg / vsegmag
        path(ell) = vortexpath[:, i] .+ ell .* vsegtan
        xi(ell) = fieldpoint .- path(ell)
        core(ell) = vortexcore[i] .+ ell .* (vortexcore[i+1] .- vortexcore[i]) / vsegmag
        function integrand(ell, params...)
            xiell = xi(ell)
            xiellmag = norm(xiell)
            c = core(ell)
            dir = cross(vsegtan, xiell)
            w = weightfn(xiellmag / c)
            return (w / xiellmag^3) .* dir
        end
        rnt_vel .+= circulation .* integrator(integrand, 0, vsegmag) ./ (4*pi)
    end

    return rnt_vel
end


#==== Straight Vortex Test Case ====#
#=
**Straight Vortex Test Case**
This is the start of a unit test of the
weighted_biot_savart_kernel function.

Let a infinitely long straight vortex be aligned with
the x-axis. We will evaluate the velocity at points
along the y-axis. 
=#
using Plots
vp = [[-1000, 0, 0] [1000, 0, 0]]  # [0 1; 0 0; 0 0] same same
vc = [0.1, 0.1]
y = zeros(Float64, 10)
y .= collect(0:2:18)  # end point is included in range
y[1] = 1e-3  # avoid divide by zero
fp = zeros(3, 10)
fp[2, :] = y

# Initialize velocity array
vel = zeros(10)
# Loop through the field points and calculate the velocity
for i in axes(fp, 2)
    # vel[i] = weighted_biot_savart_kernel(fp[:, i], vp, vc, x->1, sciml_integrals)[3]
    vel[i] = weighted_biot_savart_kernel(fp[:, i], vp, vc, x->1, sciml_integrals)[3]
end

vel_true = 1 ./ (2 .* pi .* y)

# println("vel = ", vel)
# println("vel_true = ", vel_true)
println("Error (straight vortex): ", norm(vel - vel_true))

plt = plot(y, vel, markershape=:x, label="Numerical")
plot!(y, vel_true, label="Analytical")
display(plt)
