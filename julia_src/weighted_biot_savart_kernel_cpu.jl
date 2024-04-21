using StaticArrays: SVector

include("src/biot_savart_solver.jl")


function _weighted_biot_savart_kernel_cpu!(rtnvel, fps, vpps, cdms, circs)
    # Loop over the field points
    for idx in axes(fps, 2)
        # Get a field point from the batch
        @inbounds fp = SVector{3, Float32}(
            fps[1, idx],
            fps[2, idx],
            fps[3, idx])
        
        velocity = velocity_at_field_point_bs(fp, vpps, cdms, circs)

        @inbounds rtnvel[:, idx] .= velocity
    end

    return nothing
end


################### USER API ###################

function bs_solve_cpu(fieldpoints, vorpathpoints, cordiameters, circulations)
    # Create an array to store the return velocities
    ret_vels = Array{Float32}(undef, 3, size(fieldpoints, 2))

    _weighted_biot_savart_kernel_cpu!(
        ret_vels,
        fieldpoints,
        vorpathpoints,
        cordiameters,
        circulations)

    return Array(ret_vels)
end


# ####### Run #######
# # # Benchmark 10_000 field points
# # # CUDA: 7.461 ms, 118.67 KB, 41 allocs
# # # CPU: 130.879 ms, 237.08 KB, 33 allocs
# # # @benchmark CUDA.@sync bs_solve($fps, $vpps, $vcrs, $cirs; device="cuda")
# # @benchmark Threads.@sync bs_solve($fps, $vpps, $vcrs, $cirs; device="cpu")
