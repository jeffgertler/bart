__all__ = ["fit_lightcurve", "BART"]

import numpy as np
# import scipy.optimize as op
# from scipy.signal.spectral import lombscargle
import emcee

import _bart


def quad_ld(g1, g2, r):
    onemmu = 1 - np.sqrt(1 - r * r)
    return 1 - g1 * onemmu - g2 * onemmu * onemmu


def nl_ld(c, r):
    mu = np.sqrt(1 - r ** 2)
    return 1 - sum([c[i] * (1.0 - mu ** (0.5 * (i + 1)))
                                          for i in range(len(c))])


def fit_lightcurve(t, f, ferr, rs=1.0, p=None, a=0.01, T=1.0):
    # Deal with masked and problematic data points.
    inds = ~(np.isnan(t) + np.isnan(f) + np.isnan(ferr)
           + np.isinf(t) + np.isinf(f) + np.isinf(ferr)
           + (t < 0) + (f < 0) + (ferr <= 0))
    t, f, ivar = t[inds], f[inds], 1.0 / ferr[inds] / ferr[inds]

    # Make a guess at the initialization.
    fs, iobs = np.median(f), 0.0
    e, phi, i = 0.001, np.pi, 0.0

    # Estimate the planetary radius from the bottom of the lightcurve.
    # Make sure to convert to Jupiter radii.
    if p is None:
        p = rs * np.sqrt(1.0 - np.min(f) / fs) / 9.94493e-2

    # Initialize the system.
    ps = BART(rs, fs, iobs, ldp=[50, 0.4, 0.1], ldptype="quad")
    ps.add_planet(p, a, e, T, phi, i)
    ps._data = (t, f, ivar)

    # Check the parameter conversions.
    p0 = ps.to_vector()
    ps.from_vector(p0)
    np.testing.assert_almost_equal(p0, ps.to_vector())

    # Optimize.
    # p1 = op.minimize(ps.chi2, p0)
    # ps.from_vector(p1.x)
    # return ps

    # Sample.
    ndim, nwalkers = len(p0), 100
    initial_position = [p0 * (1 + 0.1 * np.random.randn(ndim))
                                                for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ps)
    pos, lnprob, state = sampler.run_mcmc(initial_position, 100)
    sampler.reset()
    sampler.run_mcmc(pos, 100, lnprob0=lnprob)
    return sampler


class BART(object):

    def __init__(self, rs, fs, iobs, ldp=None, ldptype=None):
        self._data = None

        self.rs = rs
        self.fs = fs
        self.iobs = iobs

        self.rp, self.ap, self.ep, self.tp, self.php, self.ip = \
                                            [np.array([]) for i in range(6)]

        if ldp is not None and ldptype is None:
            self.r = ldp[:, 0]
            self.Ir = ldp[:, 1]
        elif ldptype == "quad":
            assert ldp is not None and len(ldp) == 3, \
                "You must provide the limb-darkening coefficients."
            dr = 1.0 / ldp[0]
            self.r = np.arange(0, 1, dr) + dr
            self.Ir = quad_ld(ldp[1], ldp[2], self.r)
        elif ldptype == "nl":
            assert ldp is not None and len(ldp) >= 2, \
                "You must provide the limb-darkening coefficients."
            dr = 1.0 / ldp[0]
            self.r = np.arange(0, 1, dr) + dr
            self.Ir = nl_ld(ldp[1:], self.r)
        else:
            raise Exception("You must choose a limb-darkening law.")

    def add_planet(self, r, a, e, T, phi, i):
        self.rp = np.append(self.rp, r)
        self.ap = np.append(self.ap, a)
        self.ep = np.append(self.ep, e)
        self.tp = np.append(self.tp, T)
        self.php = np.append(self.php, phi)
        self.ip = np.append(self.ip, i)

    def to_vector(self):
        v = [np.log(self.fs)]
        for i in range(len(self.rp)):
            v += [np.log(self.rp[i]),
                  np.log(self.ap[i]),
                  self.ep[i],
                  np.log(self.tp[i]),
                  self.php[i],
                  self.ip[i]]
        return np.array(v)

    def from_vector(self, v):
        self.fs = np.exp(v[0])

        npars = 6
        for j, i in enumerate(range(0, len(self.rp) * npars, npars)):
            self.rp[j] = np.exp(v[i + 1])
            self.ap[j] = np.exp(v[i + 2])
            self.ep[j] = v[i + 3]
            self.tp[j] = np.exp(v[i + 4])
            self.php[j] = v[i + 5]
            self.ip[j] = v[i + 6]

        # Check the eccentricity.
        assert np.all(self.ep >= 0) and np.all(self.ep <= 1)

    def chi2(self, p):
        assert self._data is not None
        try:
            self.from_vector(p)
        except AssertionError:
            return 1e100
        model = self.lightcurve()
        delta = self._data[1] - model
        c2 = np.sum(delta * delta * self._data[2])
        return c2

    def __call__(self, p):
        return -0.5 * self.chi2(p)

    def lightcurve(self, t=None):
        if t is None:
            assert self._data is not None
            t = self._data[0]
        return _bart.lightcurve(t, self.rs, self.fs, self.iobs,
                                self.rp, self.ap, self.ep, self.tp, self.php,
                                self.ip, self.r, self.Ir)
