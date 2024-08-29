


# Function for picking out path segments from
# a path given a single index. This returns the
# elements as StaticArrays.
# getseg()::Tuple{SVector{3, T1}, SVector{3, T1}, T1, T1, T1, T1}
function packseg(
    vpps::AbstractArray{T, 2},
    crads::AbstractArray{T, 1},
    circs::AbstractArray{T, 1},
    indx::Integer)::SVector{10, T} where {T<:AbstractFloat}

    # Get starting point of segment
    vpp1 = SVector{3, T}(
    vpps[1, indx],
    vpps[2, indx],
    vpps[3, indx])

    # Get the ending point of segment
    vpp2 = SVector{3, T}(
    vpps[1, indx+1],
    vpps[2, indx+1],
    vpps[3, indx+1])

    # @inbounds return vpp1, vpp2, crads[indx], crads[indx+1], circs[indx], circs[indx+1]
    return SVector{10, T}(vpp1..., vpp2..., crads[indx], crads[indx+1], circs[indx], circs[indx+1])
end


# Function for picking a field point out of a
# 3xN array of field points
function get3col(mat::AbstractArray{T}, indx::Integer) where {T<:AbstractFloat}
    return SVector{3, T}(mat[1, indx], mat[2, indx], mat[3, indx])
end


function compstepsize(
                    cradI::T,
                    cradF::T,
                    stepscalar::T,
                    minstepsize::T)::T where {T<:AbstractFloat}

    mincradstep = stepscalar * T(min(cradI, cradF))
    stepsize = max(mincradstep, minstepsize)

    return stepsize
end


###################### Tensor algebra functions ######################
#=======================================
Let `a` and `b` are arbitrary 3x3 matrices
which represent a rank 2 tensor, and where
`v` and `w` are arbitrary 3-element column
vectors which represent rank 1 tensors.
Julia has defined the following vector-matrix
math operations (the LinearAlgebra package
was used)
-----DOT PRODUCT-----
dot(v, w) = dot(w, v) = tranpose(v) * w = sum(v .* w)
dot(v, a) = ERROR
dot(a, v) = ERROR
dot(a, b) = sum(a .* b)
-----CROSS PRODUCT-----
cross(v, w)[i] = works as expected
cross(v, a)[i, j] = ERROR
cross(a, v)[i, j] = ERROR
cross(a, b)[i, j, k] = ERROR
-----MATRIX MULTIPLICATION-----
v * a = ERROR
a * transpose(v) = ERROR
(transpose(v) * a)[1, i] = dot(v, a[:, i])
(a * v)[i, 1] = dot(a[i, :], v)
(a * b)[i, j] = dot(a[i, :], b[:, j])
-----NOTES-----
(v .* a)[:, j] = v .* a[:, j]
(tranpose(v) .* a)[i, :] = v .* a[i, :]
=======================================#

# TensorDotVector
# return = a[i, :] * v
function LinearAlgebra.dot(A::SMatrix{3, 3}, v::SVector{3})::SVector{3}
    return SVector{3}(
        v[1] * A[1, 1] + v[2] * A[1, 2] + v[3] * A[1, 3],
        v[1] * A[2, 1] + v[2] * A[2, 2] + v[3] * A[2, 3],
        v[1] * A[3, 1] + v[2] * A[3, 2] + v[3] * A[3, 3]
    )

end

# VectorDotTensor
# return = tranpose(v) * a[i, :]
function LinearAlgebra.dot(v::SVector{3}, A::SMatrix{3, 3})::SVector{3}
    return SVector{3}(
        v[1] * A[1, 1] + v[2] * A[2, 1] + v[3] * A[3, 1],
        v[1] * A[1, 2] + v[2] * A[2, 2] + v[3] * A[3, 2],
        v[1] * A[1, 3] + v[2] * A[2, 3] + v[3] * A[3, 3]
    )
end

# TensorCrossVector
# return[i, :] = cross(a[i, :], v)
function LinearAlgebra.cross(A::SMatrix{3, 3}, v::SVector{3})::SMatrix{3, 3}
    # Note: SMatrix arranges arguments
    # in column first order
    return SMatrix{3, 3}(
        A[1, 2] * v[3] - A[1, 3] * v[2],
        A[2, 2] * v[3] - A[2, 3] * v[2],
        A[3, 2] * v[3] - A[3, 3] * v[2],
        A[1, 3] * v[1] - A[1, 1] * v[3],
        A[2, 3] * v[1] - A[2, 1] * v[3],
        A[3, 3] * v[1] - A[3, 1] * v[3],
        A[1, 1] * v[2] - A[1, 2] * v[1],
        A[2, 1] * v[2] - A[2, 2] * v[1],
        A[3, 1] * v[2] - A[3, 2] * v[1]
    )
end

# VectorCrossTensor
# return[:, i] = cross(v, a[:, i])
function LinearAlgebra.cross(v::SVector{3}, A::SMatrix{3, 3})::SMatrix{3, 3}
    # Note: SMatrix arranges arguments
    # in column first order
    return SMatrix{3, 3}(
        v[2] * A[3, 1] - v[3] * A[2, 1],
        v[3] * A[1, 1] - v[1] * A[3, 1],
        v[1] * A[2, 1] - v[2] * A[1, 1],
        v[2] * A[3, 2] - v[3] * A[2, 2],
        v[3] * A[1, 2] - v[1] * A[3, 2],
        v[1] * A[2, 2] - v[2] * A[1, 2],
        v[2] * A[3, 3] - v[3] * A[2, 3],
        v[3] * A[1, 3] - v[1] * A[3, 3],
        v[1] * A[2, 3] - v[2] * A[1, 3]
    )
end