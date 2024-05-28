#===============================================
This script implements a CPU version of the
weighted Biot-Savart solver. The solver loops
over all field points and returns a velocity
vector of langth equal to the number of field
points.
The reason this file contains the code for
looping through field points is to allow for
parallelized. The way the CPU has to be
parallelized is different from the GPU, so we
have separate implementations for each. 
===============================================#


###### Import modules and local scrips ######
using StaticArrays: SVector
include("src/weighted_biot_savart_solver/weighted_biot_savart_for_one_field_point.jl")


###### Function ######
function weighted_biot_savart_solver_cpu(fieldpoints,
                                            vorpathpoints,
                                            cordiameters,
                                            circulations;
                                            verbose=false)
    
    # Create an array to store the return velocities
    returnarray = Array{Float32}(undef, 3, size(fieldpoints, 2))

    # TIMING
    if verbose
        include(string(pwd(), "/julia_src/src/utility_functions/utility_functions.jl"))
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
            # ANSI escape codes "\e[1K", "\e[1G", https://en.wikipedia.org/wiki/ANSI_escape_code#Description
            println("  * Field point: ", fpindx, "/", size(fieldpoints, 2), "... ")
            t1 = time_ns()
        end

        velocity = weighted_biot_savart_for_one_field_point(fp,
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

    return Array(returnarray)
end
