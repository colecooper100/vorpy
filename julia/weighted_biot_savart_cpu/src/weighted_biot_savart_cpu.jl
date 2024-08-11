#===============================================
This module implements the CPU version of the
weighted Biot-Savart solver. The solver loops
over all the field points supplied to the
method and returns a vector of velocities 
equal in langth to the number of field
points supplied.
The method for computing the velocity at a
single field point given a vortex path is
implemented to run on the CPU or GPU. This,
file handles the parallelization of the method
(should any be available).
===============================================#

module weighted_biot_savart_cpu

using Base.Threads
using StaticArrays
using weighted_biot_savart_solver
using utilities

export u_wbs_cpu

function u_wbs_cpu(fieldpoints::AbstractArray{T, 2},
                                            vorpathpoints::AbstractArray{T, 2},
                                            cordradii::AbstractArray{T, 1},
                                            circulations::AbstractArray{T, 1};
                                            stepsizescalar::T=T(0.25),
                                            threaded::Bool=true) where {T<:AbstractFloat}
    # t0 = time_ns()  # TIMING
    # Convert the input arrays to StaticArrays
    sfieldpoints = SMatrix{size(fieldpoints)..., T}(fieldpoints)
    svorpathpoints = SMatrix{size(vorpathpoints)..., T}(vorpathpoints)
    scordradii = SVector{length(cordradii), T}(cordradii)
    scirculations = SVector{length(circulations), T}(circulations)
    # println("Conversion to StaticArrays: ", (time_ns() - t0) / 1e9)  # TIMING


    # Create an array to store the return velocities
    rtnvel = Matrix{T}(undef, size(sfieldpoints)...)
    # Loop over the field points
    # t0 = time_ns()  # TIMING
    if threaded
        @threads for fpindx in axes(sfieldpoints, 2)
            # println(threadid())  # DEBUG
            # Get a field point from the batch
            fp = getfp(sfieldpoints, fpindx)

            # t0 = time_ns()  # TIMING
            velocity = u_wbs_1fp(fp,
                                    svorpathpoints,
                                    scordradii,
                                    scirculations,
                                    T(stepsizescalar))
            # println("Single field point (threaded): ", (time_ns() - t0) / 1e9)  # TIMING

            @inbounds rtnvel[:, fpindx] .= velocity
        end
    else
        for fpindx in axes(sfieldpoints, 2)
            # println(threadid())  # DEBUG
            
            # Get a field point from the batch
            fp = getfp(sfieldpoints, fpindx)

            # t0 = time_ns()  # TIMING
            velocity = u_wbs_1fp(fp,
                                    svorpathpoints,
                                    scordradii,
                                    scirculations,
                                    T(stepsizescalar))
            # println("Single field point: ", (time_ns() - t0) / 1e9)  # TIMING

            @inbounds rtnvel[:, fpindx] .= velocity
        end
    end
    # println("Loop over field points: ", (time_ns() - t0) / 1e9)  # TIMING

    # return Array(returnarray)
    return rtnvel
end

end # module weighted_biot_savart_cpu
