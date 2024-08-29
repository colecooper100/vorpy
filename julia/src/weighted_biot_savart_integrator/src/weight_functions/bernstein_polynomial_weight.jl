# Define the weight function such that
# t \in [0, 1]
function bernstein_polynomial_weight(delta::T)::Tuple{T, T} where {T<:AbstractFloat}
    # De Casteljau's algorithm, unrolled, and fitted to
    # compute the weight function of Biot-Savart for
    # vortical flows.
    # For De Casteljau's algorithm, delta \in [0, 1]
    # Here, delta is proportional to the magnitude of
    # the spherical radius (m) and the core radius (a).
    # We want delta to equal 1 when the distance from
    # the core is sufficient enough that we can use 
    # the non-weighted Biot-Savart law. 
    # After some testing, Dustin determined that a
    # reasonable distance would be 5x the core size.
    # So, delta = m / (5*a)
    if delta < 0  # delta negative
        throw(string("Input for the weight function was $(delta). Input should be between 0 and 1."))
        return T(0), T(0)  #, T(0)
    elseif delta > 5  
        # Force the weight to be 1
        # This is when we are using the analytical solution
        # to the Biot-Savart law.
        return T(1), T(0)  #, T(0)
    else
        t = delta / T(5)
        ot = T(1) - t
        B_00_00 = T(0.0)
        B_00_01 = T(0.0)
        B_00_02 = T(-0.0001906562398041414)
        B_00_03 = T(0.15443423712286783)
        B_00_04 = T(0.5778731722300601)
        B_00_05 = T(1.3280659878802887)
        B_00_06 = T(0.6816157084385458)
        B_00_07 = T(1.1568940522619577)
        B_00_08 = T(0.9441585717561716)
        B_00_09 = T(1.0082051488031003)
        B_00_10 = T(1.0)
        B_00_11 = T(1.0)
        B_00_12 = T(1.0)
        B_01_00 = B_00_00 * ot + B_00_01 * t
        B_01_01 = B_00_01 * ot + B_00_02 * t
        B_01_02 = B_00_02 * ot + B_00_03 * t
        B_01_03 = B_00_03 * ot + B_00_04 * t
        B_01_04 = B_00_04 * ot + B_00_05 * t
        B_01_05 = B_00_05 * ot + B_00_06 * t
        B_01_06 = B_00_06 * ot + B_00_07 * t
        B_01_07 = B_00_07 * ot + B_00_08 * t
        B_01_08 = B_00_08 * ot + B_00_09 * t
        B_01_09 = B_00_09 * ot + B_00_10 * t
        B_01_10 = B_00_10 * ot + B_00_11 * t
        B_01_11 = B_00_11 * ot + B_00_12 * t
        B_02_00 = B_01_00 * ot + B_01_01 * t
        B_02_01 = B_01_01 * ot + B_01_02 * t
        B_02_02 = B_01_02 * ot + B_01_03 * t
        B_02_03 = B_01_03 * ot + B_01_04 * t
        B_02_04 = B_01_04 * ot + B_01_05 * t
        B_02_05 = B_01_05 * ot + B_01_06 * t
        B_02_06 = B_01_06 * ot + B_01_07 * t
        B_02_07 = B_01_07 * ot + B_01_08 * t
        B_02_08 = B_01_08 * ot + B_01_09 * t
        B_02_09 = B_01_09 * ot + B_01_10 * t
        B_02_10 = B_01_10 * ot + B_01_11 * t
        B_03_00 = B_02_00 * ot + B_02_01 * t
        B_03_01 = B_02_01 * ot + B_02_02 * t
        B_03_02 = B_02_02 * ot + B_02_03 * t
        B_03_03 = B_02_03 * ot + B_02_04 * t
        B_03_04 = B_02_04 * ot + B_02_05 * t
        B_03_05 = B_02_05 * ot + B_02_06 * t
        B_03_06 = B_02_06 * ot + B_02_07 * t
        B_03_07 = B_02_07 * ot + B_02_08 * t
        B_03_08 = B_02_08 * ot + B_02_09 * t
        B_03_09 = B_02_09 * ot + B_02_10 * t
        B_04_00 = B_03_00 * ot + B_03_01 * t
        B_04_01 = B_03_01 * ot + B_03_02 * t
        B_04_02 = B_03_02 * ot + B_03_03 * t
        B_04_03 = B_03_03 * ot + B_03_04 * t
        B_04_04 = B_03_04 * ot + B_03_05 * t
        B_04_05 = B_03_05 * ot + B_03_06 * t
        B_04_06 = B_03_06 * ot + B_03_07 * t
        B_04_07 = B_03_07 * ot + B_03_08 * t
        B_04_08 = B_03_08 * ot + B_03_09 * t
        B_05_00 = B_04_00 * ot + B_04_01 * t
        B_05_01 = B_04_01 * ot + B_04_02 * t
        B_05_02 = B_04_02 * ot + B_04_03 * t
        B_05_03 = B_04_03 * ot + B_04_04 * t
        B_05_04 = B_04_04 * ot + B_04_05 * t
        B_05_05 = B_04_05 * ot + B_04_06 * t
        B_05_06 = B_04_06 * ot + B_04_07 * t
        B_05_07 = B_04_07 * ot + B_04_08 * t
        B_06_00 = B_05_00 * ot + B_05_01 * t
        B_06_01 = B_05_01 * ot + B_05_02 * t
        B_06_02 = B_05_02 * ot + B_05_03 * t
        B_06_03 = B_05_03 * ot + B_05_04 * t
        B_06_04 = B_05_04 * ot + B_05_05 * t
        B_06_05 = B_05_05 * ot + B_05_06 * t
        B_06_06 = B_05_06 * ot + B_05_07 * t
        B_07_00 = B_06_00 * ot + B_06_01 * t
        B_07_01 = B_06_01 * ot + B_06_02 * t
        B_07_02 = B_06_02 * ot + B_06_03 * t
        B_07_03 = B_06_03 * ot + B_06_04 * t
        B_07_04 = B_06_04 * ot + B_06_05 * t
        B_07_05 = B_06_05 * ot + B_06_06 * t
        B_08_00 = B_07_00 * ot + B_07_01 * t
        B_08_01 = B_07_01 * ot + B_07_02 * t
        B_08_02 = B_07_02 * ot + B_07_03 * t
        B_08_03 = B_07_03 * ot + B_07_04 * t
        B_08_04 = B_07_04 * ot + B_07_05 * t
        B_09_00 = B_08_00 * ot + B_08_01 * t
        B_09_01 = B_08_01 * ot + B_08_02 * t
        B_09_02 = B_08_02 * ot + B_08_03 * t
        B_09_03 = B_08_03 * ot + B_08_04 * t
        B_10_00 = B_09_00 * ot + B_09_01 * t
        B_10_01 = B_09_01 * ot + B_09_02 * t
        B_10_02 = B_09_02 * ot + B_09_03 * t
        B_11_00 = B_10_00 * ot + B_10_01 * t
        B_11_01 = B_10_01 * ot + B_10_02 * t
        # return B_11_00 * ot + B_11_01 * t 
        sol = B_11_00 * ot + B_11_01 * t  # The value of the weight at t
        dsol = (B_11_01 - B_11_00) * delta * T(2.4)  # The value of the dweight/dt at t
        # d2sol = (B_10_02 - 2*B_10_01 + B_10_00) * delta^2 * T(5.28)
        return sol, dsol  #, d2sol
    end
end
