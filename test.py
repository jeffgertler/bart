import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl
import pyfits

from bart import BART


def get_model(params):
    f0, p, T, a, phi, incl = np.atleast_1d(params) ** 2
    ps = BART(10.0, 0.0)
    ps.add_planet(p, T, incl, phi, 0.0, a)
    return ps


def chi2(params, t, f, ferr):
    ps = get_model(params)
    c = np.sum(((f - ps.lightcurve(t, f0=params[0] ** 2)) / ferr) ** 2)
    print c, params
    return c


if __name__ == "__main__":
    f = pyfits.open("data/kepler4b.fits")
    lc = np.array(f[1].data)
    f.close()

    t = lc["TIME"]
    f, ferr = lc["PDCSAP_FLUX"], lc["PDCSAP_FLUX_ERR"]

    inds = ~np.isnan(t) * ~np.isnan(f)
    t, f, ferr = t[inds], f[inds], ferr[inds]

    # p0 = np.sqrt([np.median(f), 1.0, 3.2135, 100, np.pi, 0.0])
    # p1 = op.fmin_bfgs(chi2, p0, args=(t, f, ferr))
    # print p1

    pars = [405.0963342, 0.49460494, 1.79265574, 8.02959651, 1.83049726, 0]
    ps = get_model(pars)

    pl.plot(t % 3.2135, f, ".k")
    pl.plot(t % 3.2135, ps.lightcurve(t, f0=pars[0] ** 2), "+r")
    pl.savefig("kepler4b.png")
