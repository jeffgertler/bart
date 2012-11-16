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


def uniform_disks(p, b):
    return 1 - _occulted_area(1, p, b)[0] / np.pi


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
