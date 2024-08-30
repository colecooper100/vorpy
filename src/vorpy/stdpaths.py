import numpy as np
import vorpy as vp

def velfp_lamboseen(fps, vppI, vppF, crad, circ):
    """
    # Arguments
    fps: 3 array or 3xN array
    vppI: 3 array
    vppF: 3 array
    crad: float
    circ: float
    """
    segvec = vppF - vppI
    seglen = np.linalg.norm(segvec)
    segdir = segvec / seglen
    RI = fps - vppI
    RIlen = np.linalg.norm(RI, axis=1)
    s = np.sqrt(RIlen**2 - np.dot(RI, segdir)**2)
    scl = circ / (2 * np.pi * s)
    return scl * (1 - np.exp(-s**2 / (2 * crad**2)))


class linepath(vp.vorpath):
    """
    linepath: A straight line vortex path
    """

    def __init__(self, linelen, numsegs, dtyp, crad=5, circ=1):
        """
        Construct a vortex line path.

        # Arguments
        linelen: float
        numsegs: int
        dtyp: data type
        crad: float
        circ: float
        """
        self.vppF = np.array([linelen/2, 0.0, 0.0], dtype=dtyp)
        self.vppI = -self.vppF.copy()
        vpps = np.linspace(self.vppI, self.vppF, numsegs+1, dtype=dtyp)
        crads = np.ones(numsegs+1, dtype=dtyp) * dtyp(crad)
        circs = np.ones(numsegs+1, dtype=dtyp) * dtyp(circ)
        super().__init__(vpps, crads, circs)

    def velfp_lamboseen(self, fps):
        # velfp_lamboseen(fps, vppI, vppF, crad, circ)
        return velfp_lamboseen(fps, self.vppI, self.vppF, self.crads[0], self.circs[0])


class ringpath(vp.vorpath):
    """
    ringpath: A circular vortex path
    """

    def __init__(self, radius, numsegs, dtyp, crad=1, circ=1):
        """
        Construct a vortex ring path.

        For now, the ring is centered at the origin and lies
        in the x-y plane.

        # Arguments
        radius: float
        numsegs: int
        dtyp: data type
        crad: float
        circ: float
        """
        self.ringrad = radius
        xvec = radius * np.cos(np.linspace(0, 2*np.pi, numsegs+1))
        yvec = radius * np.sin(np.linspace(0, 2*np.pi, numsegs+1))
        vpps = np.zeros((numsegs+1, 3), dtype=dtyp)
        vpps[:,0] = xvec
        vpps[:,1] = yvec
        vpps[-1] = vpps[0]
        crads = np.ones(numsegs+1, dtype=dtyp) * dtyp(crad)
        circs = np.ones(numsegs+1, dtype=dtyp) * dtyp(circ)
        super().__init__(vpps, crads, circs)

        # The velocity of an infinitesimally thin vortex
        # ring on the vortex line is constant, thus we
        # compute it once when the ring is created.
        beta = 0.558
        scl = self.circs[0] / (4 * np.pi * self.ringrad)
        trm = np.log(8 * self.ringrad / (np.sqrt(2) * self.crads[0]))
        self.vel_onring_infthin = scl * (trm - beta)


            
            

