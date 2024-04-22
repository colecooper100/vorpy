using StaticArrays: SVector


include("src/weighted_biot_savart.jl")


function weighted_biot_savart_solver_cpu(# returnarray,
                                            fieldpoints,
                                            vorpathpoints,
                                            cordiameters,
                                            circulations;
                                            verbose=false)
    
    # Create an array to store the return velocities
    returnarray = Array{Float32}(undef, 3, size(fieldpoints, 2))

    # TIMING
    if verbose
        include("/home/user1/Dropbox/code/vorpy/julia_src/src/timing.jl")
        println("Inside weighted_biot_savart_solver_cpu...")
        println("* Looping over all field points...")
        # Initial time for entire execution
        t0 = time_ns()
    end
    # Loop over the field points
    for fpindx in axes(fieldpoints, 2)
        # Get a field point from the batch
        @inbounds fp = SVector{3, Float32}(
            fieldpoints[1, fpindx],
            fieldpoints[2, fpindx],
            fieldpoints[3, fpindx])
        
        # TIMING
        if verbose
            #"\e[1K", "\e[1G", 
            println("  * Field point: ", fpindx, "/", size(fieldpoints, 2), "... ")
            t1 = time_ns()
        end

        velocity = _weighted_biot_savart(fp,
                                            vorpathpoints,
                                            cordiameters,
                                            circulations;
                                            verbose=verbose)

        # TIMING
        if verbose
            println("* Done ", elapsed_time(t1), " seconds.")
        end

        @inbounds returnarray[:, fpindx] .= velocity
    end

    # TIMING
    if verbose
        println("* Total elapsed time: ", elapsed_time(t0), " seconds")
        println("Leaving weighted_biot_savart_solver_cpu.")
    end

    # return nothing
    return Array(returnarray)
end

# DELETE
# ################### USER API ###################

# function bs_solve_cpu(fieldpoints,
#                         vorpathpoints,
#                         cordiameters,
#                         circulations;
#                         verbose=false)
#     # Create an array to store the return velocities
#     ret_vels = Array{Float64}(undef, 3, size(fieldpoints, 2))

#     _weighted_biot_savart_solver_cpu!(
#         ret_vels,
#         fieldpoints,
#         vorpathpoints,
#         cordiameters,
#         circulations;
#         verbose=verbose)

#     return Array(ret_vels)
# end
