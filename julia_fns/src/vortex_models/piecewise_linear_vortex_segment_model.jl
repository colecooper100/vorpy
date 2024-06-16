using StaticArrays: SVector
using LinearAlgebra: norm


#=============================================
---------------Segment Models----------------
I wanted the vortex path model to be a function
whichs takes as input the arc length variable
and returns propteries of the vortex at that
arc length (if for no other reason then this
would match the way the problem is defined.)

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
    results in an arc length greater than the
    input arc length. This means computing a
    while loop many times when computing the
    integral. Which seemd like unwanted overhead.

The length of the path segment is not known
beforehand. I have thought of the following to
overcome this issue:
- have a separate function which computes the
    of the segment. I don't like this because
    to compute the length of a segment, which
    is somewhat complicated (more so than a straight
    line) would require doing a path integral
    anyway.
- assume ell \in [0, 1] and let the model handel
    the real arc length. The problem with this
    is that I want to take a specific stepsize
    when computing the integral and this method
    cannot do that (you would need to know the
    true length of the segment to translate [0, 1]
    to [0, seglen]).
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

Input:
    - ell: arc length variable.
        0 <= ell <= total_length
    - vpp1: start of 
        segment
    - vpp2: end of segment
    - vcr1: core diameter at start of segment
    - vcr2: core diameter at end of segment
    - vcir1: circulation at start of segment
    - vcir2: circulation at end of segment
    - params: array, model parameters
Output:
    - C: vortex path point at ell
    - vtan: unit tangent vector at ell
    - vcr: core diameter at ell
    - cir: circulation at ell
    - ell: arc length variable which resulted
        in the vortex path point (needed because
        supplied ell might be greater than the
        segment length)
    - endofseg: boolean, true if the end of the
        segment was reached
=============================================#
function piecewise_linear_vortex_segment_model(ell::T,
                                                vpp1::SVector{3, T},
                                                vpp2::SVector{3, T},
                                                crad1::T,
                                                crad2::T,
                                                circ1::T,
                                                circ2::T,
                                                params=nothing) where {T<:AbstractFloat}
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