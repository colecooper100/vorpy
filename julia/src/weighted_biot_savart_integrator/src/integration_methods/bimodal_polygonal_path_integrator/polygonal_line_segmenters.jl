# using LinearAlgebra: norm, dot

# function whole_line_segmenter(cutoffrad::T,
#                                 fp::SVector{3, T},
#                                 vpp1::SVector{3, T},
#                                 vpp2::SVector{3, T}) where {T<:AbstractFloat}

#     #===================================
#     -If the vector magnitude from any of
#     the path points to the field point
#     is less than the cutoff radius, then
#     the field point is inside the cutoff.
#     - If the field point is inbetween
#     the path points and the perpendicular
#     distance from the field point to the
#     path is less than the cutoff radius,
#     then the field point is inside the
#     cutoff.
#     ===================================#
    




# end


#============================================
Note 1: I wasn't able to construct new arrays
on the GPU (e.g., [vpp1 vpp2] throws an error).
But I could construct tuples of arrays. Which
is why the path segments are returned as tuples.

Note 2: At most, there is one path segment that
is inside the cutoff radius and two path segments
outside.

Note 3: The GPU did not like that the return
was different depending on the system. So, it
needs to return the same number and type of
values. For pathsegincutoff, I have choosen
to return Tuple{Bool, Tuple{SVector{3, T}, SVector{3, T}}}
where the Bool is true if path of the path is
inside the cutoff. For pathsegoutcutoff, I
use Tuple{Tuple{Bool, Tuple{SVector{3, T}, SVector{3, T}}}, Tuple{Bool, Tuple{SVector{3, T}, SVector{3, T}}}}.
Again, the Bool is true if the set of path
points represent a true segment. The idea being
that the boolean can be used in a look to determine
if the path points need to be processed.
The way I look at this is (ISUSED, (SEGSTART, SEGEND)).
(This was not my first choice, I decided to
do things this way to get everything to work
on the GPU.)
Example return: if the field point is before
vpp1, and some of the path is inside the
cutoff, then pathsegincutoff = (true, (vpp1, vpp1 .+ (cutendtan .* pathtan)))
and pathsegoutcutoff = ((true, (vpp1 .+ (cutendtan .* pathtan), vpp2), (false, vpp1, vpp1)).
The convention I will use for a set of unused
path points is (false, (vpp1, vpp1)).
============================================#
function precise_segmenter(cutoffrad::T,
                                    fp::SVector{3, T},
                                    vpp1::SVector{3, T},
                                    vpp2::SVector{3, T}) where {T<:AbstractFloat}
    # Compute path vector, length, and tangent
    pathvec = vpp2 .- vpp1
    pathlen = norm(pathvec)
    pathtan = pathvec ./ pathlen
    
    # Compute xi1
    # xi1 is the vector that extends from vpp1 to
    # the field point
    xi1 = fp .- vpp1

    # Compute the components of xi1, where pathtan is
    # one axis, and the other two axes are \hat z and
    # perpendicular to the line axis. We get the
    # component tangent to the path by projecting xi1
    # onto the tangent axis (i.e., the dot product of
    # xi1 with pathtan).
    xi1tan = dot(xi1, pathtan)
    # To get the component perpendicular to the path
    # axis, we subtract the component tangent to the
    # path axis from xi1 and then take the norm.
    xi1perp = norm(xi1 .- xi1tan * pathtan)

    # To determine if the filed point is inside of
    # the cutoff radius we consider the right triangle
    # where the hypotenuse is the cutoff radius and
    # the opposite side is xi1perp. As long as xi1perp
    # is less than or equal to cutoffrad, the field 
    # point must be inside the cutoff radius. But,
    # because the cutoff radius is fixed, another
    # way to determine if the field point is inside
    # the cutoff radius (which can be more definite
    # when evaluating) is to consider the adjacent
    # side.
    # - If xi1perp = cutoffrad, then adjlen = 0.
    # - If xi1perp = 0, then adjlen = cutoffrad.
    # - If 0 < xi1perp < cutoffrad, then 0 < adjlen < cutoffrad.
    # - If xi1perp > cutoffrad, then adjlen < 0.
    # However, because we need to use a square root
    # to compute adjlen, we need to ensure that the
    # argument to the square root is non-negative.
    # To do this we take the maximum of 0 and the
    # the difference between the cutoffrad^2 and
    # xi1perp^2.
    # Additionally, the length of the adjacent side
    # is used to determine how much of the path is
    # inside the cutoff radius
    # The maximum possible amount of path that will
    # be cut cannot be more than 2*adjlen).    
    adjlen = sqrt(max(0, cutoffrad^2 - xi1perp^2))
    if adjlen > 0  # Field point is inside cutoff
        # Using adjlen we determine the possible
        # start and end points of the path segment.
        # To do this, we need only consider the
        # component that is tangent to the path axis.
        cutstarttan = xi1tan - adjlen
        cutendtan = xi1tan + adjlen
        # println("cutstarttan = ", cutstarttan)  # DEBUG
        # println("cutendtan = ", cutendtan)  # DEBUG

        #==================================================
        Determine what part of the path (if any)
        needs to be cut. There are five cases to
        consider:
        - If the end of the cut segment is before vpp1,
        then the entire path is outside of the cutoff.
        - If the start of the cut segment is after vpp2,
        then the entire path is outside of the cutoff.
        - If the start of the cut segment is after vpp1
        and the end of the cut segment is before vpp2,
        then path of the path is inside the cutoff, with
        the ends of the path being outside of the cutoff.
        - If the start of the cut segment is before vpp1,
        and the end of the cut is before vpp2, then only
        the start of the path is inside the cutoff.
        - If the start of the cut segment is after vpp1,
        and the end of the cut segment is after vpp2, then
        only the end of the path is inside the cutoff.
        ==================================================#
        # Tolerance for floating point comparison
        tol = 5f-6  
        # If -tol <= number <= tol, then number is zero
        # If pathlen - tol <= number <= pathlen + tol, then number is pathlen
        if cutendtan <= tol || pathlen - tol <= cutstarttan
            # Cut segment is entirely outside of the path
            pathsegincutoff = (false, (vpp1, vpp1))
            pathsegoutcutoff = ((true, (vpp1, vpp2)), (false, (vpp1, vpp1)))
        elseif cutstarttan <= tol && pathlen - tol <= cutendtan
            # Cut segment is entirely inside the path
            pathsegincutoff = (true, (vpp1, vpp2))
            pathsegoutcutoff = ((false, (vpp1, vpp1)), (false, (vpp1, vpp1)))
        elseif cutstarttan < tol
            # Cut segment starts at vpp1
            pathsegincutoff = (true, (vpp1, vpp1 .+ (cutendtan .* pathtan)))
            pathsegoutcutoff = ((true, ((vpp1 .+ (cutendtan .* pathtan), vpp2))), (false, (vpp1, vpp1)))
        elseif pathlen - tol <= cutendtan
            # Cut segment ends at vpp2
            pathsegincutoff = (true, (vpp1 .+ (cutstarttan .* pathtan), vpp2))
            pathsegoutcutoff = ((true, (vpp1, vpp1 .+ (cutstarttan .* pathtan))), (false, (vpp2, vpp2)))
        else
            # Cut segment is between vpp1 and vpp2
            pathsegincutoff = (true, (vpp1 .+ (cutstarttan .* pathtan), vpp1 .+ (cutendtan .* pathtan)))
            pathsegoutcutoff = ((true, (vpp1, (vpp1 .+ (cutstarttan .* pathtan)))), (true, ((vpp1 .+ (cutendtan .* pathtan)), vpp2)))
        end
    else
        # Field point outside cutoff
        pathsegincutoff = (false, (vpp1, vpp1))
        pathsegoutcutoff = ((true, (vpp1, vpp2)), (false, (vpp1, vpp1)))
    end
    return pathsegincutoff, pathsegoutcutoff
end
