using StaticArrays: SVector
using LinearAlgebra: norm

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

#=============================================
---------------Segment Models----------------
I wanted to setup the model of the vortex path
as a function that took the arc length variable
and returned propteries of the vortex at that
point. If for no other reason then this would
match the way the problem is defined.
Some of the problems I encounted when setting
up a model which takes arc length as Input:
- Integration is done along the entire leng9th
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
function piecewise_linear_vortex(ell, vpp1, vpp2, vcr1, vcr2, cir1, cir2, params=nothing)
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
        vpp = vpp2
        vcr = vcr2
        cir = cir2
        rtnell = seglen
        endofseg = true
    else
        # Using the model, compute the path
        # point
        vpp = vpp1 .+ (ell .* vtan)
        # Using the model, compute the core
        # diameter
        vcr = vcr1 + (ell / seglen) * (vcr2 - vcr1)
        # Using the model, compute the circulation
        cir = cir1 + (ell / seglen) * (cir2 - cir1)
        rtnell = ell
        endofseg = false
    end
    return vpp, vtan, vcr, cir, rtnell, endofseg
end




# ################### DEBUGGING ###################
# using CUDA

# #==============================================
# -----------Straight Vortex Test Case-----------
# We define a Lamb-Oseen vortex aligned with the
# z-axis.

# If our coordinate system has +z running from
# left-to-right, then +x is into the page and +y
# runs from down-to-up. We compute the flow
# velocity at points along the +x-axis at the
# origin, i.e., $\vec r = (x, 0, 0)$.
# ==============================================#
# # Set problem parameters
# VDOMAIN = [-1000, 1000]  # Vortex path domain
# FPDOMAIN = [0, 20]  # Field points domain
# NUMVPSEGS = 3  # Number of vortex path segments
# NUMFP = 5  # Number of field points
# println("VDOMAIN = ", VDOMAIN)
# println("FPDOMAIN = ", FPDOMAIN)
# println("NUMVPSEGS = ", NUMVPSEGS)
# println("NUMFP = ", NUMFP)

# # Define the vortex
# vpps = zeros(Float32, 3, NUMVPSEGS + 1)  # Path points
# vpps[1, :] .= range(VDOMAIN[1], stop=VDOMAIN[2], length=NUMVPSEGS + 1)
# vcrs = Float32.(collect(axes(vpps, 2)))  # Core diameters
# cirs = Float32.(collect(axes(vpps, 2)))  # Circulation

# # Define the field points
# x = zeros(Float32, NUMFP)
# x .= range(FPDOMAIN[1], FPDOMAIN[2], length=NUMFP)  # end point is included in range
# if x[1] == 0 
#     x[1] = 1e-3  # avoid divide by zero
# end
# fps = zeros(Float32, 3, NUMFP)
# fps[1, :] .= x


# ###### Run on the CPU ######
# ELL = Float32(1900)
# rtncpu = piecewise_linear_vortex(
#     ELL,
#     vpps[:, 1],
#     vpps[:, 2],
#     vcrs[1],
#     vcrs[2],
#     cirs[1],
#     cirs[2])
# println("CPU: ", rtncpu)

# ###### Run on the GPU ######
# # Make a wrapper to run functions on the GPU
# function gpu_wrapper(rtn, ell, vpp1, vpp2, vcr1, vcr2, cir1, cir2)
#     svpp1 = SVector{3, Float32}(vpp1[1], vpp1[2], vpp1[3])
#     svpp2 = SVector{3, Float32}(vpp2[1], vpp2[2], vpp2[3])
#     vpp, segtan, vcr, cir, rtnell, endofseg = piecewise_linear_vortex(ell, svpp1, svpp2, vcr1, vcr2, cir1, cir2)
#     rtn[:, 1] .= vpp
#     rtn[:, 2] .= segtan
#     rtn[1, 3] = vcr
#     rtn[2, 3] = cir
#     rtn[3, 3] = rtnell
#     return nothing
# end

# cuvpps = CuArray(vpps)
# cuvpp1 = cuvpps[:, 1]
# cuvpp2 = cuvpps[:, 2]
# cuvcrs = CuArray(vcrs)
# cuvcr1 = vcrs[1]
# cuvcr2 = vcrs[2]
# cucirs = CuArray(cirs)
# cucir1 = cirs[1]
# cucir2 = cirs[2]
# cufps = CuArray(fps)
# curtn = CuArray{Float32}(undef, 3, 3)

# # @device_code_warntype
# # @device_code_llvm
# kern = @cuda launch=false gpu_wrapper(curtn, ELL, cuvpp1, cuvpp2, cuvcr1, cuvcr2, cucir1, cucir2)
# # @cuda gpu_wrapper(curtn, ELL, cuvpp1, cuvpp2, cuvcr1, cuvcr2, cucir1, cucir2)

# println("Max threads: ", CUDA.maxthreads(kern))
# println("Register usage: ", CUDA.registers(kern))
# println("Memory usage: ", CUDA.memory(kern))

# # println("curtn = ")  # DEBUG
# # display(curtn)  # DEBUG