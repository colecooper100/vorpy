using CUDA
using StaticArrays
using LinearAlgebra

include("vortex_models.jl")


#=============================================
I don't think CUDA supports nested functions,
so, functions used in the kernel must be
defined as stand alone functions (outside of
the kernel).

Note: The code doesn't have to be physically
outside of the kernel, you just can't assume you
have access to any variables local to the outer
function.
=============================================#


#################### BS integrand ####################

function bsintegrand(fp, vppnts, vcrads, vcircs, sindx, ell)
    xiell, vtanell = xi(fp, vppnts, sindx, ell)
    xiellmag = norm(xiell)
    corell = vcoremodel(vcrads, sindx, ell)
    circell = vcircmodel(vcircs, sindx, ell)
    direll = cross(vtanell, xiell)
    weightell = bsweightmodel((xiellmag / corell)^2)
    println("xiell = ", xiell)  # DEBUG
    println("vtanell = ", vtanell)  # DEBUG
    println("xiellmag = ", xiellmag)  # DEBUG
    println("corell = ", corell)  # DEBUG
    println("circell = ", circell)  # DEBUG
    println("direll = ", direll)  # DEBUG
    println("weightell = ", weightell)  # DEBUG
    return (weightell * circell / xiellmag^3) .* direll
end


#################### Numerical BS integrators ####################

# Define BS solver
function bs_uniform_trapezoidal_rule(numsteps, fp, vppnts, vcrads, vcircs, sindx)
    stepsize = Float32(1) / numsteps
    if stepsize < 1e-6
        throw(string("Percision Error: integrator stepsize is: ", stepsize))
        # @cuprintln("Warning: integrator stepsize is: ", stepsize)
    else
        sol = bsintegrand(fp, vppnts, vcrads, vcircs, sindx, Float32(0))
        # println("sol_1 = ", sol)  # DEBUG
        sol = sol .+ bsintegrand(fp, vppnts, vcrads, vcircs, sindx, Float32(1))
        # println("sol_2 = ", sol)  # DEBUG
        sol = sol .* Float32(0.5)
        # println("sol_3 = ", sol)  # DEBUG

        # Start stepindex at 2 because we already did
        # the first step and use 'less-than' becase we
        # already did the last step
        stepindex = UInt32(2)
        while stepindex < numsteps
            sol = sol .+ bsintegrand(fp, vppnts, vcrads, vcircs, sindx, stepindex * stepsize)
            # println("sol_$(stepindex + 2) = ", sol)  # DEBUG
            stepindex += UInt32(1)
        end
    end

    # println("sol_$(stepindex + 2) = ", sol)  # DEBUG

    # @cuprintln("typeof(sol): ", typeof(sol))  # DEBUG
    return sol .* (stepsize / Float32(4 * pi))
end