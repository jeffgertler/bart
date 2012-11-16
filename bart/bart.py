__all__ = ["fit_lightcurve", "BART"]

import numpy as np
import emcee

import _bart
import triangle


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

        self._nplanets = 0
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

    @property
    def nplanets(self):
        return self._nplanets

    def add_planet(self, r, a, e, T, phi, i):
        self._nplanets += 1
        self.rp = np.append(self.rp, r)
        self.ap = np.append(self.ap, a)
        self.ep = np.append(self.ep, e)
        self.tp = np.append(self.tp, T)
        self.php = np.append(self.php, phi)
        self.ip = np.append(self.ip, i)

    def to_vector(self):
        n = self.nplanets
        v = []

        if "fs" in self._par_list:
            v += [np.log(self.fs)]

        if "r" in self._par_list:
            v += [np.log(self.rp[i]) for i in range(n)]

        if "T" in self._par_list:
            v += [np.log(self.tp[i]) for i in range(n)]

        if "phi" in self._par_list:
            v += [np.log(self.php[i]) for i in range(n)]

        if "i" in self._par_list:
            v += [self.ip[i] for i in range(n)]

        return np.array(v)

    def from_vector(self, v):
        ind, n = 0, self.nplanets

        if "fs" in self._par_list:
            self.fs = np.exp(v[ind])
            ind += 1

        if "r" in self._par_list:
            self.rp = np.exp(v[ind:ind + n])
            ind += n

        if "T" in self._par_list:
            self.tp = np.exp(v[ind:ind + n])
            ind += n

        if "phi" in self._par_list:
            self.php = np.exp(v[ind:ind + n])
            ind += n

        if "i" in self._par_list:
            self.ip = np.array(v[ind:ind + n])
            ind += n

        return self

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

    def fit(self, t, f, ferr, pars=["T", "r", "phi"]):
        # Deal with masked and problematic data points.
        inds = ~(np.isnan(t) + np.isnan(f) + np.isnan(ferr)
            + np.isinf(t) + np.isinf(f) + np.isinf(ferr)
            + (t < 0) + (f < 0) + (ferr <= 0))
        t, f, ivar = t[inds], f[inds], 1.0 / ferr[inds] / ferr[inds]

        # Store the data.
        self._data = [t, f, ferr]

        # Fitting parameters.
        self._par_list = pars

        # Check vector conversions.
        p0 = self.to_vector()
        self.from_vector(p0)
        assert np.all(p0 == self.to_vector())

        # Set up emcee.
        nwalkers, ndim = 50, len(p0)
        self._sampler = emcee.EnsembleSampler(nwalkers, ndim, self)

        # Sample the parameters.
        p0 = emcee.utils.sample_ball(p0, 0.001 * p0, size=nwalkers)
        pos, lprob, state = self._sampler.run_mcmc(p0, 500)
        print lprob
        # self._sampler.reset()
        # self._sampler.run_mcmc(pos, 1000, lnprob0=lprob)

        # Let's see some stats.
        print("Acceptance fraction: {0:.2f} %"
                .format(np.mean(self._sampler.acceptance_fraction)))

        return self._sampler.flatchain

    def plot_fit(self, bp=""):
        import matplotlib.pyplot as pl

        assert self._data is not None and self._sampler is not None, \
                "You have to fit some data first."

        chain = self._sampler.flatchain
        print np.exp(np.median(chain, axis=0))

        triangle.corner(np.exp(chain.T), labels=self._par_list, bins=50)
        pl.savefig("parameters.pdf")

        for i in range(chain.shape[1]):
            pl.clf()
            pl.plot(self._sampler.chain[:, :, i].T)
            pl.savefig("time.{0}.png".format(i))

        # lc_fig = pl.figure(figsize=(6, 8))

        # # Generate light curve samples.
        # t = np.linspace(0, T, 500)
        # f = np.empty((len(chain), len(t)))
        # for ind, v in enumerate(chain):
        #     f[ind, :] = system.from_vector(v).lightcurve(t)
        # f = f.T
