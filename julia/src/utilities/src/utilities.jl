module utilities

using StaticArrays

# for timing use time_ns()

export getfp, getseg

# Function for picking a field point out of a
# 3xN array of field points
function getfp(fps::AbstractArray{T1, 2}, indx::T2) where {T1<:AbstractFloat, T2<:Integer}
    @inbounds return SVector{3, T1}(fps[1, indx], fps[2, indx], fps[3, indx])
end

# Function for picking out path segments from
# a path given a single index. This returns the
# elements as StaticArrays.
# getseg()::Tuple{SVector{3, T1}, SVector{3, T1}, T1, T1, T1, T1}
function getseg(vpps::AbstractArray{T1, 2},
    crads::AbstractArray{T1, 1},
    circs::AbstractArray{T1, 1},
    indx::T2) where {T1<:AbstractFloat, T2<:Integer}

    # Get starting point of segment
    @inbounds vpp1 = SVector{3, T1}(
    vpps[1, indx],
    vpps[2, indx],
    vpps[3, indx])

    # Get the ending point of segment
    @inbounds vpp2 = SVector{3, T1}(
    vpps[1, indx+1],
    vpps[2, indx+1],
    vpps[3, indx+1])

    @inbounds return vpp1, vpp2, crads[indx], crads[indx+1], circs[indx], circs[indx+1]
end

end # module utilities
