using CUDA  # DEBUG
using StaticArrays: SVector
using LinearAlgebra: norm

include("weight_function.jl")


#=============================================
I don't think CUDA supports nested functions,
so, the functions used in the kernel must be
defined as stand alone functions (outside of
the kernel).

Note: The code doesn't have to be physically
outside of the kernel, you just can't assume you
have access to any variables local to the outer
function.
=============================================#


#============================================== 
---------------Vortex Model---------------
Input:
    - ell: arc length variable.
        0 <= ell <= total_length
    - vpps: vortex path points
    - params: array, model parameters
Output:
    - vortex path point at ell
    - unit tangent vector at ell
    - core diameter at ell
    - circulation at ell
==============================================#
function piecewise_linear_path(ell, vpps, vcrs, cirs, params=nothing)
    numpts = UInt32(size(vpps, 2))
    
    # Get the correct POINT index (not segment)
    pnt1 = SVector{3, Float32}(0, 0, 0)
    seg = SVector{3, Float32}(0, 0, 0)
    seglength = Float32(0)
    indx = UInt32(1)
    pathlength = Float32(0)
    while pathlength <= ell && indx < numpts
        # Get starting point of segment
        @inbounds pnt1 = SVector{3, Float32}(
            vpps[1, indx],
            vpps[2, indx],
            vpps[3, indx])

        # Get segment
        @inbounds seg = SVector{3, Float32}(
            vpps[1, indx+1] - pnt1[1],
            vpps[2, indx+1] - pnt1[2],
            vpps[3, indx+1] - pnt1[3])

        # Get segment length
        seglength = norm(seg)

        # ADVANCE THE LOOP!!!
        pathlength = pathlength + seglength
        indx = indx + UInt32(1)
    end

    if indx >= numpts && abs(pathlength - ell) > 1f-3
        # Specified arc length $\ell$ is
        # greater than the total path length
        # throw("The specified arc length $(ell) is greater than the total path length $(pathlength)")
        return nothing
    else
        # posinseg = ell - (pathlength - seglength)
        # # println("posinseg = ", posinseg)  # DEBUG
        # # Get the correct path point
        # vtan = seg ./ seglength
        # vpp = pnt1 .+ (posinseg .* vtan)
        # # println("pnt1 = ", pnt1)  # DEBUG
        # # println("vpp = ", vpp)  # DEBUG

        # # Core model simple linear interpolation
        # # (posinseg / seglength) should be between 0 and 1
        # vcr = vcrs[indx-1] + (posinseg / seglength) * (vcrs[indx] - vcrs[indx-1])

        # # Circulation model simple linear interpolation
        # cir = cirs[indx-1] + (posinseg / seglength) * (cirs[indx] - cirs[indx-1])

        # return vpp, vtan, vcr, cir
        return 2
    end
end


# Set problem parameters
VDOMAIN = [-1000, 1000]  # Vortex path domain
FPDOMAIN = [0, 20]  # Field points domain
NUMVPSEGS = 3  # Number of vortex path segments
NUMFP = 5  # Number of field points
println("VDOMAIN = ", VDOMAIN)
println("FPDOMAIN = ", FPDOMAIN)
println("NUMVPSEGS = ", NUMVPSEGS)
println("NUMFP = ", NUMFP)

# Define the vortex
## Path points
vpps = zeros(Float32, 3, NUMVPSEGS + 1)
vpps[1, :] .= range(VDOMAIN[1], stop=VDOMAIN[2], length=NUMVPSEGS + 1)
## Core diameters
vcrs = Float32.(collect(axes(vpps, 2)))  # ones(Float32, NUMVPSEGS + 1)
## Circulation
cirs = Float32.(collect(axes(vpps, 2)))  # ones(Float32, NUMVPSEGS + 1)

# Define the field points
x = zeros(Float32, NUMFP)
x .= range(FPDOMAIN[1], FPDOMAIN[2], length=NUMFP)  # end point is included in range
if x[1] == 0 
    x[1] = 1e-3  # avoid divide by zero
end
fps = zeros(Float32, 3, NUMFP)
fps[1, :] .= x
# println("fps = ")  # DEBUG
# display(fps)  # DEBUG



function kernel_function(rtn, ell, vpps, vcrs, cirs)
    rtn .= piecewise_linear_path(ell, vpps, vcrs, cirs)
    return nothing
end

cuell = Float32(1000)
cuvpps = CuArray(vpps)
cuvcrs = CuArray(vcrs)
cucirs = CuArray(cirs)
rtn = CuArray{Float32}(undef, 3)

# @device_code_warntype
# @device_code_llvm
@device_code_warntype @cuda kernel_function(rtn, cuell, cuvpps, cuvcrs, cucirs)

println("rtn = ", rtn)  # DEBUG

# function vcoremodel(vcrads, indx, ell)
#     @inbounds return 
# end


# function vcircmodel(vcircs, indx, ell)
#     @inbounds return 
# end





# function xi(fp, vppnts, indx, ell)
#     vpell, vtanell = vpathmodel(vppnts, indx, ell)
#     return fp .- vpell, vtanell
# end



# # # No weight function
# # bsweightmodel(x) = Float32(1)

# # Bernstein polynomial weight function
# bsweightmodel(x) = bernstein_polynomial_weight(x)