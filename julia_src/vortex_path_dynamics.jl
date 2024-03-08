



#===========================================
Solve:
    \dot u (\vec r) = \frac{\Gamma}{4 \pi} \int_C \frac{\hat t \times \vec \xi}{\|\vec \xi|^3} dl
    where \vec \xi = \vec r - \vec C(l) and \hat t = \frac{d \vec C}{dl}

It's a coupled first-order ODE system. So,
code up an ODE solver and then use the BS
law to compute velocity.

I think it would be easy enough to extend
the BS law code to include the ODE solver
for the vortex path.

Todos:
- [ ] Code up an ODE solver
- [ ] Compute the velocity at a field point
    with the `velocity_field_point` function
    where the field points are the vortex
    path points.
===========================================#