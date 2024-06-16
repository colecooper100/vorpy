import numpy as np

# Generate vortex points for ...

def vortex_ring(num_segs, position_center, major_radius=1e0):
    theta = np.linspace(0, 2 * np.pi, num_segs + 1)
    vpx = major_radius * np.cos(theta) + position_center[0]
    vpy = major_radius * np.sin(theta) + position_center[1]
    vpz = np.zeros_like(vpx) + position_center[2]
    vpps = np.concatenate([vpx[:, np.newaxis], vpy[:, np.newaxis], vpz[:, np.newaxis]], axis=1)
    # To prevent numerical errors, explicitly set
    # the last point to the the first.
    vpps[-1] = vpps[0]
    return vpps

def three_fold_loop(num_segs, position_center, radius=1e0, shape_factor=3e-1):
    # If shape_factor is 0, then the loop is a circle
    # with radius `radius`.
    theta = np.linspace(0, 2 * np.pi, num_segs + 1)
    cylin_radius = radius * (1 + shape_factor * np.cos(3 * theta))
    vpx = cylin_radius * np.cos(theta) + position_center[0]
    vpy = cylin_radius * np.sin(theta) + position_center[1]
    vpz = np.zeros_like(vpx) + position_center[2]
    vpps = np.concatenate([vpx[:, np.newaxis], vpy[:, np.newaxis], vpz[:, np.newaxis]], axis=1)
    # To prevent numerical errors, explicitly set
    # the last point to the the first.
    vpps[-1] = vpps[0]
    return vpps

# def trefoil_knot(num_segs, position_center):