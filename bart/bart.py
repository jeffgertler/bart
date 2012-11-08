import numpy as np
import scipy.optimize as op


def compute_occulted_area(r0, p, b):
    b = np.atleast_1d(np.abs(b))
    r0 = np.atleast_1d(r0)
    ret = np.zeros([len(r0), len(b)])

    b = b[None, :] * np.ones_like(ret)
    r0 = r0[:, None] * np.ones_like(ret)

    p2 = p * p
    b2 = b * b
    r2 = r0 * r0

    m2 = (np.abs(r0 - p) < b) * (b < r0 + p)
    m3 = b <= r0 - p
    m4 = b <= p - r0

    # Fully occulted and un-occulted.
    ret[m3] = np.pi * p2
    ret[m4] = np.pi * r2[m4]

    b = b[m2]
    b2 = b2[m2]
    r0 = r0[m2]
    r2 = r2[m2]

    k1 = np.arccos(0.5 * (b2 + p2 - r2) / b / p)
    k2 = np.arccos(0.5 * (b2 + r2 - p2) / b / r0)
    k3 = np.sqrt((p + r0 - b) * (b + p - r0) * (b - p + r0) * (b + r0 + p))

    ret[m2] = p2 * k1 + r2 * k2 - 0.5 * k3

    return ret


def histogram_limb_darkening(p, b, r, Ir):
    r2 = r * r
    norm = np.pi * (r2[0] * Ir[0] + np.sum(Ir[1:] * (r2[1:] - r2[:-1])))

    areas = compute_occulted_area(r, p, b)
    dA = areas[1:] - areas[:-1]

    return 1 - (areas[0] * Ir[0] + np.sum(Ir[1:, None] * dA, axis=0)) / norm


def quad_ld(g1, g2, r):
    onemmu = 1 - np.sqrt(1 - r * r)
    return 1 - g1 * onemmu - g2 * onemmu * onemmu


class Planet(object):

    def __init__(self, radius, period, incl, phi, e, a):
        self.radius = radius
        self.T = period
        self.incl = incl
        self.phi = phi
        self.e = e
        self.a = a

    def eccentric_anomaly(self, t):
        """
        Numerically solve for the eccentric anomaly as a function of time.

        """
        e = self.e
        wt = 2 * np.pi * np.atleast_1d(t) / self.T + self.phi
        psi0s = wt + e * np.sin(wt)
        f = lambda psi, wt0: wt0 - psi + e * np.sin(psi)
        return np.array(map(lambda (psi0, wt0): op.newton(f, psi0,
                                                args=(wt0,)), zip(psi0s, wt)))

    def coords(self, t, i0=0.0):
        psi = self.eccentric_anomaly(t)
        cpsi = np.cos(psi)

        e = self.e
        d = 1 - e * cpsi

        cth = (cpsi - e) / d
        r = self.a * d

        # In the plane of the orbit.
        x, y = r * cth, np.sign(np.sin(psi)) * r * np.sqrt(1 - cth * cth)

        # Rotate into observable coordinates.
        i = i0 + self.incl
        return x * np.cos(i), y, x * np.sin(i)


class BART(object):

    def __init__(self, radius, incl):
        self.radius = radius
        self.incl = incl
        self.planets = []

        dr = 0.01
        self.r = np.arange(0, 1, dr) + dr
        self.Ir = quad_ld(0.5, 0.1, self.r)

    def add_planet(self, r, T, i, phi, e, a):
        r0 = self.radius
        self.planets.append(Planet(r / r0, T, i, phi, e, a / r0))

    def lightcurve(self, t):
        lc = np.ones_like(t)
        for p in self.planets:
            x, y, z = p.coords(t, i0=self.incl)
            m = x > 0
            b = np.sqrt(y[m] ** 2 + z[m] ** 2)
            lc[m] *= histogram_limb_darkening(p.radius, b, self.r, self.Ir)

        return lc

if __name__ == '__main__':
    import matplotlib.pyplot as pl

    np.random.seed(42)
    ps = BART(10.0, -1.3)

    # Hot Jupiter.
    # T = 0.05
    # ps.add_planet(1.2, T, 1.3, np.pi, 0.01, 108)

    # Jupiter.
    ps.add_planet(1.0, 12.0, 1.3, np.pi, 0.05, 11000)

    # Saturn.
    # ps.add_planet(0.85, 30.0, 2.5, 0.0, 0.05, 21000)

    t = 150.0 * np.random.rand(84091)
    lc = ps.lightcurve(t)

    pl.plot(t % 12, lc, ".k")
    pl.xlim(5.99, 6.01)
    pl.savefig("lightcurve.png")
