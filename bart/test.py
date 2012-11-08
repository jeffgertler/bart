import numpy as np
from scipy.integrate import quad, dblquad


def _compute_single_occ(p, b, Ir):
    # Un-occulted.
    if 1 + p <= b:
        return 0.0

    # Pre-compute the squares.
    p2 = p * p
    b2 = b * b

    if 1 <= b < p + 1:
        cth0 = 0.5 * (b2 + p2 - 1) / b / p
        rmin = lambda cth: b * cth - np.sqrt(1 + b2 * (cth * cth - 1))
        rmax = lambda _: p
    elif 1 - p <= b < 1:
        cth0 = -1.0
        rmin = lambda _: 0
        rmax = lambda cth: min(p, b * cth + np.sqrt(1 + b2 * (cth * cth - 1)))
    else:
        cth0 = -1.0
        rmin = lambda _: 0
        rmax = lambda _: p

    i = lambda r, cth: r * Ir(np.sqrt(r * r + b2 - 2 * b * r * cth)) \
                                                    / np.sqrt(1 - cth * cth)
    val = 2 * dblquad(i, cth0, 1, rmin, rmax)[0]

    return val


def brute_force(p, b, Ir):
    norm = 2 * np.pi * quad(lambda r: r * Ir(r), 0, 1)[0]
    o = np.array([_compute_single_occ(p, b0, Ir) for b0 in b])
    return 1 - o / norm


def _occulted_area(r0, p, b):
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


def uniform_disks(p, b):
    return 1 - _occulted_area(1, p, b)[0] / np.pi


def histogram_limb_darkening(p, b, r, Ir):
    r2 = r * r
    norm = np.pi * (r2[0] * Ir[0] + np.sum(Ir[1:] * (r2[1:] - r2[:-1])))

    areas = _occulted_area(r, p, b)
    dA = areas[1:] - areas[:-1]

    return 1 - (areas[0] * Ir[0] + np.sum(Ir[1:, None] * dA, axis=0)) / norm


def quad_ld(g1, g2):
    def ld(r):
        onemmu = 1 - np.sqrt(1 - r * r)
        return 1 - g1 * onemmu - g2 * onemmu * onemmu

    return ld


if __name__ == "__main__":
    import matplotlib.pyplot as pl

    p = 0.1
    b = np.linspace(0.0, 1.2, 50)

    dr = 0.005
    r = np.arange(0, 1, dr) + dr

    ld = quad_ld(0.5, 0.1)
    Ir = ld(r)

    ld_lc = histogram_limb_darkening(p, b, r, Ir)

    F = brute_force(p, b, ld)

    pl.plot(b, F - ld_lc, "k")
    # pl.plot(b, ld_lc, "--r")

    # pl.ylim(0.98, 1.001)
    pl.savefig("brute.png")
