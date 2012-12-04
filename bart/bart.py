__all__ = ["BART", "LimbDarkening", "QuadraticLimbDarkening",
           "NonlinearLimbDarkening"]


from collections import OrderedDict
from multiprocessing import Process

import numpy as np
import matplotlib.pyplot as pl
import emcee

from . import _bart
from . import mog
from . import triangle


class BART(object):

    def __init__(self, fs, iobs, ldp):
        self._data = None

        self.fs = fs
        self.iobs = iobs

        self._nplanets = 0
        self.rp, self.ap, self.ep, self.tp, self.php, self.ip = \
                                            [np.array([]) for i in range(6)]

        self._pars = OrderedDict()

        self.ldp = ldp

        self._processes = []

    def __del__(self):
        if hasattr(self, u"_processes"):
            for p in self._processes:
                p.join()

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
            v.append(p.getter(self))

        return np.array(v)

    def from_vector(self, v):
        ind, n = 0, self.nplanets

        for i, (k, p) in enumerate(self._pars.iteritems()):
            p.setter(self, v[i])

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
        lnp = 0.0

        # Priors on the limb darkening profile.
        ldp = self.ldp

        # LDP must be strictly positive.
        if np.any(ldp.intensity < 0) or np.any(ldp.intensity > 1):
            return -np.inf

        # The gammas in the quadratic case must sum to less than one and be
        # greater than or equal to zero.
        if hasattr(ldp, "gamma1") and hasattr(ldp, "gamma2"):
            g1, g2 = ldp.gamma1, ldp.gamma2
            sm = g1 + g2
            if not 0 < sm < 1 or g1 < 0 or g2 < 0:
                return -np.inf

        if np.any(self.ep < 0) or np.any(self.ep > 1):
            return -np.inf

        return lnp

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
        return _bart.lightcurve(t, self.fs, self.iobs,
                                self.rp, self.ap, self.ep, self.tp, self.php,
                                self.ip, self.ldp.bins, self.ldp.intensity)

    def fit_for(self, *args):
        [self._fit_for(p) for p in args]

    def _fit_for(self, var):
        n = self.nplanets
        if var == "fs":
            self._pars["fs"] = Parameter(r"$f_s$", "fs")

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
                self._pars["{0}{1}".format(var, i)] = LogParameter(
                    tex.format(i + 1), attr, i)

        elif var == "ldp":
            for i in range(len(self.ldp.intensity) - 1):
                self._pars["ldp_{0}".format(i)] = LDPParameter(
                        r"$\Delta I_{{{0}}}$".format(i), ind=i)

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
        np.testing.assert_almost_equal(p0, self.to_vector())

        # Set up emcee.
        nwalkers, ndim = 200, len(p0)
        self._sampler = None
        s = emcee.EnsembleSampler(nwalkers, ndim, self, threads=10)

        # Sample the parameters.
        p0 = emcee.utils.sample_ball(p0, 0.001 * p0, size=nwalkers)

        for i in range(0):
            p0, lprob, state = s.run_mcmc(p0, 500, storechain=False)

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

            for n in np.arange(len(p0))[~inds]:
                lp = self.lnprob(p0[n])
                # NOTE: this _could_ go on forever.
                while np.isinf(lp):
                    p0[n] = np.random.multivariate_normal(mu, cov)
                    lp = self.lnprob(p0[n])

            print("Rejected {0} walkers.".format(np.sum(~inds)))
            s.reset()

        # Reset and rerun.
        s.run_mcmc(p0, 100, thin=10)

        # Let's see some stats.
        print("Acceptance fraction: {0:.2f} %"
                .format(np.mean(s.acceptance_fraction)))

        self._sampler = s
        return self._sampler.flatchain

    def plot_fit(self, true_ldp=None):
        p = Process(target=_async_plot, args=("_lc_and_ldp", self, true_ldp))
        p.start()
        self._processes.append(p)

    def _lc_and_ldp(self, true_ldp):
        chain = self._sampler.flatchain

        # Compute the best fit period.
        T = np.exp(np.median(chain[:, self._pars.keys().index("T0")]))

        # Generate light curve samples.
        t = np.linspace(0, T, 500)
        f = np.empty((len(chain), len(t)))
        ld = [self.ldp.plot()[0],
              np.empty((len(chain), 2 * len(self.ldp.bins)))]
        for ind, v in enumerate(chain):
            f[ind, :] = self.from_vector(v).lightcurve(t)
            ld[1][ind, :] = self.ldp.plot()[1]
        f = f.T
        ld[1] = ld[1].T

        # Plot the fit.
        time, flux, ivar = self._data
        pl.figure(figsize=(6, 4))
        pl.plot(t, f, "#4682b4", alpha=0.1, zorder=1)
        pl.errorbar(time % T, flux, yerr=1.0 / np.sqrt(ivar), fmt=".k",
                    zorder=2)

        pl.savefig("lc.png")

        # Plot the limb-darkening.
        pl.clf()
        pl.plot(*ld, color="#4682b4", alpha=0.1)
        if true_ldp is not None:
            pl.plot(*true_ldp, color="k", lw=2)
        pl.savefig("ld.png")

    def plot_triangle(self, truths=None):
        p = Process(target=_async_plot, args=("_triangle", self, truths))
        p.start()
        self._processes.append(p)

    def _triangle(self, truths):
        assert self._data is not None and self._sampler is not None, \
                u"You have to fit some data first."

        chain = self._sampler.flatchain

        # Plot the parameter histograms.
        plotchain = self._sampler.flatchain
        inds = []
        for i, (k, p) in enumerate(self._pars.iteritems()):
            plotchain[:, i] = p.iconv(chain[:, i])
            if u"ldp" not in k:
                inds.append(i)

        if truths is not None:
            truths = [truths.get(k)
                    for i, (k, p) in enumerate(self._pars.iteritems())
                    if i in inds]

        print np.median(plotchain[:, inds], axis=0)
        print truths

        triangle.corner(plotchain[:, inds].T, labels=[str(p)
                                for k, p in self._pars.iteritems()], bins=20,
                                truths=truths)

        pl.savefig(u"triangle.png")


def _async_plot(pltype, ps, *args):
    return getattr(ps, pltype)(*args)


class LimbDarkening(object):

    def __init__(self, bins, intensity):
        self.bins = bins
        self.intensity = intensity

    def plot(self):
        x = [0, ]
        [(x.append(b), x.append(b)) for b in self.bins]
        y = []
        [(y.append(i), y.append(i)) for i in self.intensity]

        return x[:-1], y


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

    def __init__(self, name, attr=None, ind=None):
        self.name = name
        self.attr = attr
        self.ind = ind

    def __str__(self):
        return self.name

    def conv(self, val):
        return val

    def iconv(self, val):
        return val

    def getter(self, ps):
        if self.ind is None:
            return self.conv(getattr(ps, self.attr))
        return self.conv(getattr(ps, self.attr)[self.ind])

    def setter(self, ps, val):
        if self.ind is None:
            setattr(ps, self.attr, self.iconv(val))
        else:
            getattr(ps, self.attr).__setitem__(self.ind, self.iconv(val))


class LogParameter(Parameter):

    def conv(self, val):
        return np.log(val)

    def iconv(self, val):
        return np.exp(val)


class LDPParameter(LogParameter):

    def getter(self, ps):
        return self.conv(ps.ldp.intensity[self.ind]
                         - ps.ldp.intensity[self.ind + 1])

    def setter(self, ps, val):
        j = self.ind
        ps.ldp.intensity.__setitem__(j + 1,
                                     ps.ldp.intensity[j] - self.iconv(val))
