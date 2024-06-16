#===============================================
This is a utility function for computing the
root mean square error between two arrays.

To compute the RMS error, pass the two arrays
to the function. The function returns the RMS
error and the element-wise error vector.
===============================================#

function RMSerror(arr1, arr2)
    # println("Size of arr1: ", size(arr1))  # DEBUG
    # println("Size of arr2: ", size(arr2))  # DEBUG
    errvec = arr1 .- arr2
    # println("Size of err_vec: ", size(err))  # DEBUG
    RMSerr = sqrt.(sum(errvec .^ 2) / length(errvec))
    return RMSerr, errvec
end