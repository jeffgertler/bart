import numpy as np

import _bart


def quad_ld(g1, g2, r):
    onemmu = 1 - np.sqrt(1 - r * r)
    return 1 - g1 * onemmu - g2 * onemmu * onemmu


class BART(object):

    def __init__(self, radius, incl):
        self.radius = radius
        self.incl = incl

        self.rp = []
        self.ap = []
        self.ep = []
        self.tp = []
        self.php = []
        self.ip = []

        dr = 0.01
        self.r = np.arange(0, 1, dr) + dr
        self.Ir = quad_ld(0.5, 0.1, self.r)

    def add_planet(self, r, a, e, T, phi, i):
        self.rp.append(r)
        self.ap.append(a)
        self.ep.append(e)
        self.tp.append(T)
        self.php.append(phi)
        self.ip.append(i)

    def lightcurve(self, t, f0=1.0):
        return _bart.lightcurve(t, self.radius, f0, self.incl,
                                self.rp, self.ap, self.ep, self.tp, self.php,
                                self.ip, self.r, self.Ir)

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
