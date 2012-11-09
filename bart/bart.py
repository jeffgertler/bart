import numpy as np

import _bart


def quad_ld(g1, g2, r):
    onemmu = 1 - np.sqrt(1 - r * r)
    return 1 - g1 * onemmu - g2 * onemmu * onemmu


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

    def lightcurve(self, t, f0=1.0):
        rp = [p.radius * self.radius for p in self.planets]
        ap = [p.a * self.radius for p in self.planets]
        ep = [p.e for p in self.planets]
        tp = [p.T for p in self.planets]
        php = [p.phi for p in self.planets]
        ip = [p.incl for p in self.planets]
        return _bart.lightcurve(t, self.radius, f0, self.incl,
                                rp, ap, ep, tp, php, ip,
                                self.r, self.Ir)
        # lc = f0 * np.ones_like(t)
        # for p in self.planets:
        #     x, y, z = p.coords(t, i0=self.incl)
        #     m = x > 0
        #     b = np.sqrt(y[m] ** 2 + z[m] ** 2)
        #     lc[m] *= histogram_limb_darkening(p.radius, b, self.r, self.Ir)

        # return lc

if __name__ == "__main__":
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
