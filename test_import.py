import src.vorpy.vorpy as vp
# import vorpy as vp
# import vormake as vmk
import numpy as np
import matplotlib.pyplot as plt

# vmk.vortex_ring(100, [0, 0, 0], 1)


###### Make a vortex ######
phi = np.linspace(0, 2*np.pi, 100)
r = 1 + 0.25 * np.cos(6*phi)
vppsx = r*np.cos(phi)
vppsy = r*np.sin(phi)
vpps = np.zeros((100, 3), dtype=float)
vpps[:, 0] = vppsx
vpps[:, 1] = vppsy
# Set the first and last point to be the same
# to close the vortex line
vpps[-1, :] = vpps[0, :]

# Set the core radii and circulations
crads = np.array([1, 1, 1])
circs = np.array([1, 1, 1])

# Plot the vortex line
plt.plot(vpps[:, 0], vpps[:, 1])
plt.title('Vortex in the xy-plane')
plt.xlabel('x')
plt.ylabel('y')


###### Make field points ######
fps = np.zeros((3, 3), dtype=float)
fps[0, :] = vpps[0, :]
fps[1, :] = vpps[9, :]
fps[2, :] = vpps[21, :]

# Plot the field points
plt.scatter(fps[:, 0], fps[:, 1], color='r', label='Field points')

plt.show()


###### Compute the velocity at the field points ######
vels = vp.wbs_solve(fps,
                    vpps,
                    crads,
                    circs,
                    device='cpu')

print('vels:\n', vels)