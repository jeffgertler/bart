__all__ = ["BART", "LimbDarkening", "QuadraticLimbDarkening",
           "NonlinearLimbDarkening"]


from collections import OrderedDict

import numpy as np
import emcee

import _bart
import triangle
import mog


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

    def __init__(self, rs, fs, iobs, ldp):
        self._data = None

        self.rs = rs
        self.fs = fs
        self.iobs = iobs

        self._nplanets = 0
        self.rp, self.ap, self.ep, self.tp, self.php, self.ip = \
                                            [np.array([]) for i in range(6)]

        self._pars = OrderedDict()

        self.ldp = ldp

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
        v = []

        for k, p in self._pars.iteritems():
            v.append(p.conv(p.getter(self)))

        return np.array(v)

    def from_vector(self, v):
        ind, n = 0, self.nplanets

        for i, (k, p) in enumerate(self._pars.iteritems()):
            p.setter(self, p.iconv(v[i]))

        return self

    def __call__(self, p):
        return self.lnprob(p)

    def lnprob(self, p):
        self.from_vector(p)

        lnp = self.lnprior()
        if np.isinf(lnp) or np.isnan(lnp):
            return -np.inf

        return self.lnlike() + lnp

    def lnprior(self):
        ldp = self.ldp
        g1, g2 = ldp.gamma1, ldp.gamma2
        sm = g1 + g2
        if isinstance(ldp, QuadraticLimbDarkening) and (not 0 < sm < 1
                or g1 < 0 or g2 < 0):
            return -np.inf

        if np.any(self.ep < 0) or np.any(self.ep > 1):
            return -np.inf

        return 0.0

    def lnlike(self):
        assert self._data is not None
        model = self.lightcurve()
        delta = self._data[1] - model
        chi2 = np.sum(delta * delta * self._data[2])
        return -0.5 * chi2

    def lightcurve(self, t=None):
        if t is None:
            assert self._data is not None
            t = self._data[0]
        return _bart.lightcurve(t, self.rs, self.fs, self.iobs,
                                self.rp, self.ap, self.ep, self.tp, self.php,
                                self.ip, self.ldp.bins, self.ldp.intensity)

    def fit_for(self, *args):
        [self.fit_for_parameter(p) for p in args]

    def fit_for_parameter(self, var):
        n = self.nplanets
        if var == "fs":
            getter = lambda ps: getattr(ps, "fs")
            setter = lambda ps, val: setattr(ps, "fs", val)
            self._pars["fs"] = Parameter(r"$f_s$", getter, setter,
                                        conv=np.log, iconv=np.exp)
        elif var in ["T", "r", "a", "phi"]:
            if var == "T":
                tex, attr = r"$T_{0}$", "tp"
            elif var == "r":
                tex, attr = r"$r_{0}$", "rp"
            elif var == "phi":
                tex, attr = r"$\phi_{0}$", "php"
            elif var == "a":
                tex, attr = r"$a_{0}$", "ap"

            for i in range(n):
                getter = lambda ps: getattr(ps, attr)[i]

                def setter(ps, val):
                    getattr(ps, attr)[i] = val

                self._pars["{0}{1}".format(var, i)] = Parameter(
                    tex.format(i + 1), getter, setter,
                    conv=np.log, iconv=np.exp)
        elif var in ["gamma1", "gamma2"]:
            getter = lambda ps: getattr(ps.ldp, var)
            setter = lambda ps, val: setattr(ps.ldp, var, val)
            self._pars[var] = Parameter(r"$\gamma_{0}$".format(var[-1]),
                                        getter, setter)
        else:
            raise RuntimeError("Unknown parameter {0}".format(var))

    def fit(self, t, f, ferr, pars=["fs", "T", "r", "a", "phi"]):
        # Deal with masked and problematic data points.
        inds = ~(np.isnan(t) + np.isnan(f) + np.isnan(ferr)
            + np.isinf(t) + np.isinf(f) + np.isinf(ferr)
            + (t < 0) + (f < 0) + (ferr <= 0))
        t, f, ivar = t[inds], f[inds], 1.0 / ferr[inds] / ferr[inds]

        # Store the data.
        self._data = [t, f, ivar]

        # Fitting parameters.
        self.fit_for(*pars)

        # Check vector conversions.
        p0 = self.to_vector()
        self.from_vector(p0)
        assert np.all(p0 == self.to_vector())

        # Set up emcee.
        nwalkers, ndim = 500, len(p0)
        self._sampler = emcee.EnsembleSampler(nwalkers, ndim, self)

        # Sample the parameters.
        p0 = emcee.utils.sample_ball(p0, 0.001 * p0, size=nwalkers)

        for i in range(2):
            p0, lprob, state = self._sampler.run_mcmc(p0, 100)

            # Cluster to get rid of crap.
            mix = mog.MixtureModel(4, np.atleast_2d(lprob).T)
            mix.run_kmeans()
            rs, rmx = mix.kmeans_rs, np.argmax(mix.means.flatten())

            # Choose the "good" walkers.
            inds = rs == rmx
            good = p0[inds].T

            # Sample from the multi-dimensional Gaussian.
            mu, cov = np.mean(good, axis=1), np.cov(good)
            p0[~inds] = np.random.multivariate_normal(mu, cov, np.sum(~inds))

            print("Rejected {0} walkers.".format(np.sum(~inds)))

        # Reset and rerun.
        self._sampler.reset()
        self._sampler.run_mcmc(p0, 100, thin=10)

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

        triangle.corner(chain.T, labels=[str(p)
                                for k, p in self._pars.iteritems()], bins=50)
        pl.savefig("parameters.png")

        for i in range(chain.shape[1]):
            pl.clf()
            pl.plot(self._sampler.chain[:, :, i].T)
            pl.savefig("time.{0}.png".format(i))

        # Compute the best fit period.
        print self._pars.keys()
        T = np.exp(np.median(chain[:, self._pars.keys().index("T0")]))

        # Generate light curve samples.
        t = np.linspace(0, T, 500)
        f = np.empty((len(chain), len(t)))
        for ind, v in enumerate(chain):
            f[ind, :] = self.from_vector(v).lightcurve(t)
        f = f.T

        # Plot the data.
        time, flux, ivar = self._data
        pl.figure(figsize=(6, 4))
        pl.errorbar(time % T, flux, yerr=1.0 / np.sqrt(ivar), fmt=".k")

        # Over plot the fit.
        pl.plot(t, f, "#4682b4", alpha=0.1)

        pl.savefig("lc.png")


class LimbDarkening(object):

    def __init__(self, bins, intensity):
        self.bins = bins
        self.intensity = intensity


class QuadraticLimbDarkening(LimbDarkening):

    def __init__(self, nbins, gamma1, gamma2):
        dr = 1.0 / nbins
        self.bins = np.arange(0, 1, dr) + dr
        self.gamma1, self.gamma2 = gamma1, gamma2

    @property
    def intensity(self):
        onemmu = 1 - np.sqrt(1 - self.bins * self.bins)
        return 1 - self.gamma1 * onemmu - self.gamma2 * onemmu * onemmu


class NonlinearLimbDarkening(LimbDarkening):

    def __init__(self, nbins, coeffs):
        dr = 1.0 / nbins
        self.bins = np.arange(0, 1, dr) + dr
        self.coeffs = coeffs

    @property
    def intensity(self):
        mu = np.sqrt(1 - self.bins ** 2)
        c = self.coeffs
        return 1 - sum([c[i] * (1.0 - mu ** (0.5 * (i + 1)))
                                            for i in range(len(c))])


class Parameter(object):

    def __init__(self, name, getter, setter, conv=None, iconv=None):
        self.name = name
        self.getter = getter
        self.setter = setter
        self.conv = conv if conv is not None else lambda x: x
        self.iconv = iconv if iconv is not None else lambda x: x

    def __str__(self):
        return self.name
