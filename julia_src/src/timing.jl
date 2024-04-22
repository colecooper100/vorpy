function elapsed_time(t0)
    return Int(time_ns() - t0) * 10^-9
end