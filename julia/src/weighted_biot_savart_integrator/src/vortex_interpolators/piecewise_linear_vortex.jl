# using CUDA  # For testing
# using StaticArrays  # For testing
# using LinearAlgebra  # For testing
# using utilities  # For testing

#=============================================
# Segment Interpolators
In order to integrate the Biot-Savart law
along a vortex path, we are going to need to
be able to interpolate the vortex's properties
at a given arc length. 

I decided to put the vortex path model into
its own directory after reading the performance
tips on Julia's website which strongly
encuraged using functions because they often
run faster than top level code (this has to 
do with how Julia compiles code). I'm going to
let the path model determine if the end of the
segment has been reached. This allows the
interpolator to just interpolate vortex
properties.

Because I have decided to pass the collection
of vortex properties as a single array (this
way, the number of properties being passed
can be changed easily). Thus, functions that
take all of the vortex properites as arguments
will use the vpprops array i.e., a flat array
with the following order:
[vppI, vppF, cradI, cradF, circI, circF]

The return of any interpolator is an SVector
of the vortex properties which have been
flattened into a single array. The array has
the following order: [rtncrad, rtncirc]
for a total of 2 elements.

The linear segment interpolator has the
following logic (this assumes that
Lvpp(ell2) >= Lvpp(ell1) for all ell2 > ell1
and that dot(Rvpp, t) >= 0 for all ell):
1. t = vppF .- vppI: the vector which
    extends from vppI to vppF.
2. Rvpp = vpp(ell) .- vppI: vpp
    is the point on the vortex path at the
    given arc length ell (this will likely
    be computed by a path model and then
    passed to the interpolator).
3. Lvpp = norm(dot(Rvpp, t)); Lt = norm(t);
    vorprops .* (Lvpp / Lt)
4. Return vorprops

This isn't the only way to interpolate the
vortex properties; and making this modular,
interpolations methods can be swapped in
easily.
=============================================#
#=============================================
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

# Interpolator
function piecewise_linear_vortex(
                    vpp::SVector{3, T},
                    vppI::SVector{3, T},
                    vppF::SVector{3, T},
                    cradI::T,
                    cradF::T,
                    circI::T,
                    circF::T)::Tuple{T, T} where {T<:AbstractFloat}
    
    # Compute segment vector
    segvec = vppF .- vppI
    seglen = norm(segvec)

    # Unit tagent of segment
    unttanseg = segvec ./ seglen

    Rvpp = vpp .- vppI
    Rtanmag = dot(Rvpp, unttanseg)

    # Interpolate the vortex properties
    rtncrad = cradI + ((Rtanmag / seglen) * (cradF - cradI))
    rtncirc = circI + ((Rtanmag / seglen) * (circF - circI))

    return rtncrad, rtncirc
end


# #=============================================#
# # Test

# # Make a gpu wrapper for the function
# function gpu_kernel(rtn, vpp, vpprops)
#     rtn .= test_change_name(vpp, vpprops)
#     return nothing
# end

# function test_change_name(vpp, vpprops)
#     return piecewise_linear_segments(vpp, vpprops)
# end

# TYP = Float32
# VPP = SVector{3, TYP}(-5, 0, 0)
# VPPI = SVector{3, TYP}(-10, 0, 0)
# VPPF = SVector{3, TYP}(10, 0, 0)
# CRADI = TYP(5)
# CRADF = TYP(2)
# CIRCI = TYP(1)
# CIRCF = TYP(3)

# vpprops = pacvpprops(VPPI, VPPF, CRADI, CRADF, CIRCI, CIRCF)

# cpurtn = piecewise_linear_segments(VPP, vpprops)
# println("cpurtn = ", cpurtn)

# curtn = CuArray{TYP}(undef, 2)
# # threads=1 blocks=1 
# cukern = @cuda launch=false gpu_kernel(curtn, VPP, vpprops)
# # println("curtn = ", curtn)
# println("Max threads: ", CUDA.maxthreads(cukern))  # 1024
# println("Mem used: ", CUDA.memory(cukern))  # local: 88, shared: 0, constant: 0