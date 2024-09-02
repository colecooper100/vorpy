# using CUDA  # For testing
# using StaticArrays  # For testing
# using LinearAlgebra  # For testing


#========================================
This needs to take arc length ell as an
argument and return vpp(ell), tang(ell),
ell, and endofseg.
========================================#

function linear_polygonal_path(
                vppI::SVector{3, T},
                vppF::SVector{3, T},
                ell::T)::Tuple{SVector{3, T}, SVector{3, T}, T, Bool} where {T<:AbstractFloat}
    
    # Compute segment vector
    segvec = vppF .- vppI
    seglen = norm(segvec)

    if ell >= seglen
        # At or pass end of segment
        rtnvpp = vppF
        rtnell = seglen
        rtnendofseg = true
    else
        # Not at end of segment
        rtnvpp = vppI .+ (ell / seglen) .* segvec
        rtnell = ell
        rtnendofseg = false
    end

    rtnunttanseg = segvec ./ seglen
    return rtnvpp, rtnunttanseg, rtnell, rtnendofseg

end  # function linear_polygonal_path


# #===============================================#
# # Testing
# TYP = Float32
# VPPI = SVector{3, TYP}(0, 0, 0)
# VPPF = SVector{3, TYP}(10, 0, 0)
# ell = TYP(1.5)
# cpurtn = linear_polygonal_path(VPPI, VPPF, ell)
# println("CPU Return: ", cpurtn)

# # GPU testing
# function gpu_kernel(rtnvpp, segtang, rtnell, rtnendofseg,  # return values
#                         vppI, vppF, ell)  # input values
#     linear_polygonal_path(vppI, vppF, ell)
#     # rtn = linear_polygonal_path(vppI, vppF, ell)
#     # rtnvpp .= rtn[1]
#     # segtang .= rtn[2]
#     # rtnell .= rtn[3]
#     # rtnendofseg .= rtn[4]

#     return nothing
# end

# gpurtnvpp = CuArray{TYP}(undef, 3)
# gpusegtang = CuArray{TYP}(undef, 3)
# gpurnell = CuArray{TYP}(undef, 1)
# gpurtnendofseg = CuArray{Bool}(undef, 1)

# @cuda threads=1 blocks=1 gpu_kernel(gpurtnvpp, gpusegtang, gpurnell, gpurtnendofseg, VPPI, VPPF, ell)
# print("GPU Return: (", Array(gpurtnvpp))
# print(", ", Array(gpusegtang))
# print(", ", Array(gpurnell))
# println(", ", Array(gpurtnendofseg), ")")

# cukern = @cuda launch=false gpu_kernel(gpurtnvpp, gpusegtang, gpurnell, gpurtnendofseg, VPPI, VPPF, ell)
# println(launch_configuration(cukern.fun))  # (blocks = 40, threads = 768)
# println("GPU register use: ", CUDA.registers(cukern))  # 30
# println("Max threads: ", CUDA.maxthreads(cukern))  # 1024