using StaticArrays
using LinearAlgebra


#=============================================
---------------Segment Models----------------
The (weighted) Biot-Savart integrand requires
properties of the vortex at a given arc length
along some specified path. Compartimentalize
this process so code that requires this
information does not need to 'know' how to
compute these values, it just assumes it is
getting a specific set of values from a
function defined to be a vortex model. This
should make the code modular and easy to
revise.

Some of the problems I encounted:
- Integration is done along the entire length
    of the vortex path. This means you need to
    know the total length of the vortex path
    or you need to take steps until you reach
    the end of the vortex path. (This is not
    too bad and can be done with a while loop.)
- If you don't know the length of the path and
    you don't know the length of each segment,
    then each time the function gets a new
    arc length, it has to step through the 
    the path until it (maybe) finds a segment
    which, when added to the previous segments,
    results in a step which is beyond the given
    path. This means computing using a while
    loop when computing the integral (i.e., the
    process can't be parallelized). 


The length of the path segment is not known
beforehand. I have thought of the following to
overcome this issue:
- Let the user put in any value for ell and if
    the value is greater than the segment length,
    the function returns the end of the segment,
    the value of ell at the end of the path and
    a boolean which tells the user that the end
    of the segment was reached. This is the method
    I have chosen to use.
    - The benefit of this method is that we get
        ell back, so, because the core and
        circulation must follow the vortex path
        we can feed the returned ell into those
        models and not have to make them as robust
        as this method.
=============================================#
function piecewise_linear_segments(ell::T,
                                    vpp1::SVector{3, T},
                                    vpp2::SVector{3, T},
                                    crad1::T,
                                    crad2::T,
                                    circ1::T,
                                    circ2::T) where {T<:AbstractFloat}
    # Compute length of segment (how this is done
    # will depend on the model)
    seg = vpp2 .- vpp1
    seglen = norm(seg)

    # The tagent of the segment is constant so
    # we can compute it here
    vtan = seg ./ seglen

    if ell >= seglen
        # Specified arc length $\ell$ is
        # greater than the segment length
        vpp_rtn = vpp2
        crad_rtn = crad2
        circ_rtn = circ2
        ell_rtn = seglen
        endofseg = true
    else
        # Using the model, compute the path
        # point
        vpp_rtn = vpp1 .+ (ell .* vtan)
        # Using the model, compute the core
        # diameter
        crad_rtn = crad1 + (ell / seglen) * (crad2 - crad1)
        # Using the model, compute the circulation
        circ_rtn = circ1 + (ell / seglen) * (circ2 - circ1)
        ell_rtn = ell
        endofseg = false
    end

    return vpp_rtn, vtan, crad_rtn, circ_rtn, ell_rtn, endofseg
end