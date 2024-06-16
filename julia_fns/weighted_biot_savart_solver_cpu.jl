#===============================================
This script implements a CPU version of the
weighted Biot-Savart solver. The solver loops
over all field points and returns a velocity
vector of langth equal to the number of field
points.
The reason this file contains the code for
looping through field points is to allow for
parallelization. When available, the velocity
is computed for mutliple field points at once.
The code for parallelizing on the CPU differs
from the GPU code. Also, not every system is
going to be able to run the GPU code,
However, the Julia code is being used as the
backend for a Python package, so, in Python
we use a try-except block to try to import
the GPU code, and if it fails, we import the
CPU code only.
===============================================#


###### Turn on debugging ######
DEBUG = true


###### Import modules and local scrips ######
using StaticArrays: SVector, SMatrix

# Import utility functions
if DEBUG
    include(string(UTILITY_FUNCTIONS, "/utility_functions.jl"))
    # t0 = time_ns()  # TIMING
    # elapsed_time(t0)  # TIMING
end

t0 = time_ns()  # TIMING
# Import the function that calculates the velocity
include(string(WEIGHTED_BIOT_SAVART_SOLVER_ONE_FIELD_POINT, "/weighted_biot_savart_solver_one_field_point.jl"))

if DEBUG
    println("-> Elapsed time for importing weighted_biot_savart_solver_one_field_point.jl: ", elapsed_time(t0))
    # with include statement: 0.031228281000000004
    # without include statement: 0.005935766
    # Difference: 0.025292514000000004
end

###### Function ######
function weighted_biot_savart_solver_cpu(fieldpoints::AbstractArray{T, 2},
                                            vorpathpoints::AbstractArray{T, 2},
                                            cordradii::AbstractArray{T, 1},
                                            circulations::AbstractArray{T, 1};
                                            stepsizescalar::T=T(0.5)) where {T<:AbstractFloat}

    if DEBUG
        tfnstart = time_ns()  # TIMING
    end

    # Convert the input arrays to StaticArrays
    sfieldpoints = SMatrix{size(fieldpoints)..., T}(fieldpoints)
    svorpathpoints = SMatrix{size(vorpathpoints)..., T}(vorpathpoints)
    scordradii = SVector{length(cordradii), T}(cordradii)
    scirculations = SVector{length(circulations), T}(circulations)

    if DEBUG
        println("-> Time converting user arrays to StaticArrays: ", elapsed_time(tfnstart))
    end

    # Create an array to store the return velocities
    returnarray = Array{T}(undef, size(sfieldpoints)...)
    
    debugloopcounter = UInt8(0)
    # Loop over the field points
    for fpindx in axes(sfieldpoints, 2)
        # # Get a field point from the batch
        # @inbounds fp = SVector{3, Float32}(
        #     sfieldpoints[1, fpindx],
        #     sfieldpoints[2, fpindx],
        #     sfieldpoints[3, fpindx])

        if DEBUG
            if debugloopcounter <= 3
                tloopstart = time_ns()  # TIMING
                debugloopcounter += 1
            end
        end

        # Get a field point from the batch
        # Since sfieldpoints is already a StaticArray,
        # we can just take a slice from it, and the result
        # is still a StaticArray.
        fp = sfieldpoints[:, fpindx]

        velocity = weighted_biot_savart_for_one_field_point(fp,
                                                            svorpathpoints,
                                                            scordradii,
                                                            scirculations,
                                                            T(stepsizescalar))

        returnarray[:, fpindx] .= velocity
        # @inbounds returnarray[:, fpindx] .= velocity

        if DEBUG
            if debugloopcounter <= 3
                println("-> Time computing velocity for field point ", fpindx, ": ", elapsed_time(tloopstart))
            end
        end
    end

    if DEBUG
        println("-> Time computing velocities for all field points: ", elapsed_time(tfnstart))
    end

    # return Array(returnarray)
    return returnarray
end
