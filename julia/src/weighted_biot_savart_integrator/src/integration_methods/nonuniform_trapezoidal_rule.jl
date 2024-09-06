#=================================================
This method is a general integrator of a line
integral. It is used by the bimodel integration
methods of the WBS integral when inside the cutoff
radius. I have a feeling that this method should
be general and make no assumptions about what is
passed or returned by the integrand function.

(At least at this time) more complex integrators
(like the bimodel methods) will probably need to
be a specific to the problem they are integrating.
Thus, it probably makes more sense to pass
specific parameters rather than a single vector.
(I don't believe there is much, if any, performance
penalty for doing this.)

Nonuniform trapezoidal rule
    We assume the values being integrated are
    packed flat in a single SVector. 
=================================================#

#=================================================
I don't think I will need to change the integrand
function dynamically. I think this needs to be
flexible enough so that it can be changed depending
on our needs, but I also think once things are
working as expected and have been tested, this
will not need to be changed.

I originally had this in wbs_integrand_function.jl
but I don't really think anything else will need
it other than the integrators used to solve the
WBS integral when inside a cutoff radius, and if
the integration method changes, we can copy this
integrand over to the scrip (or we change how we
are doing things at that time, but for now it
seems logical to me to put this here.
=================================================#

# Set the integrand function to the WBS integrand
# include("./integrand_functions/vel_velgrad_integrand.jl")
include("./integrand_functions/wbs_integrand_function.jl")
# Set the return type of the integrand function
function integrand(
            params::SVector{13, T},
            ell::T) where {T<:AbstractFloat}
    # rtnvec, rtngrad, endofseg = vel_velgrad_integrand(fp, vpprops, ell)
    # rtnvals = wbs_integrand_function(params, ell)  # DEBUG
    # return SVector{3, T}(1, 1, 1), rtnvals[2], rtnvals[3]  # DEBUG
    return wbs_integrand_function(params, ell)
end

# F(x) = \int_a^b f(x) dx \approx \sum_{i=1}^{N} (f(x_{i-1}) + f(x_i))/2 * \Delta x_i
# where \Delta x_i = x_i - x_{i-1}
# function nonuniform_trapezoidal_rule(rtnval::AbstractArray{T, 1}, stepsize::T, params::SVector) where {T<:AbstractFloat}  # No return
function nonuniform_trapezoidal_rule(
                                    stepsize::T,
                                    params::SVector) where {T<:AbstractFloat}
    
    # Evaluate integrand function at ell=0
    # to initialize the integrator
    curr_eval, ell, endofseg = integrand(params, T(0))

    # Initialize the return value
    # This needs 
    # It needs to have the same shape and type as
    # the return of the integrand function
    rtnval = SVector{3, T}(0, 0, 0)

    # println("stepsize: ", stepsize)  # DEBUG

    # Step through the segment
    # itercount starts at 1 because we already did one
    # evaluation of the integrand above.
    itercount = 1
    while !endofseg
        # println("rtnval (before): ", rtnval)  # DEBUG
        itercount += 1
        # Advance the method by one step
        prev_ell = ell
        prev_eval = curr_eval
        ellmaybe = prev_ell + stepsize
        # print("ellmaybe: ", ellmaybe)  # DEBUG
        # Evaluate the integrand at ellmaybe step
        curr_eval, ell, endofseg = integrand(params, ellmaybe)
        # println(" ell: ", ell)  # DEBUG
        # Acculmulate the step solutions of
        # the integrand
        deltaell = ell - prev_ell
        rtnval = rtnval .+ (((prev_eval .+ curr_eval) ./ T(2)) .* deltaell)
        # print("prev_eval: ", prev_eval)  # DEBUG
        # print(" curr_eval: ", curr_eval)  # DEBUG
        # print(" prev_eval .+ curr_eval: ", prev_eval .+ curr_eval)  # DEBUG
        # print(" deltaell: ", deltaell)  # DEBUG
        # println(" (((prev_eval .+ curr_eval) ./ T(2)) .* deltaell): ", (((prev_eval .+ curr_eval) ./ T(2)) .* deltaell))  # DEBUG
        # println("rtnval (after): ", rtnval)  # DEBUG
    end

    # # DEBUG
    # println("itercount: ", itercount)

    return rtnval, itercount
end