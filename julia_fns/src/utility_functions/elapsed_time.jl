#===============================================
This is a utility function for timing the 
execution of code.

To perform timing store the initial time like
`t0 = time_ns()`. Then to get the elapsed time
call `elapsed_time(t0)`. The function returns
the elapsed time in seconds.
===============================================#

function elapsed_time(t0)
    return Int(time_ns() - t0) * 10^-9
end