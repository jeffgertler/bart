from __future__ import print_function

__all__ = [u"BART", u"LimbDarkening", u"QuadraticLimbDarkening",
           u"NonlinearLimbDarkening"]


from collections import OrderedDict
from multiprocessing import Process
import cPickle as pickle

import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl
import h5py
import emcee

from . import _bart
from . import mog
from . import triangle


class BART(object):

    def __init__(self, fs, iobs, ldp, jitter=0.01):
        self._data = None

        self.fs = fs
        self.iobs = iobs
        self.jitter = jitter

        self._nplanets = 0
        self.rp, self.ap, self.ep, self.tp, self.php, self.pop, self.ip = \
                                            [np.array([]) for i in range(7)]

        self._pars = OrderedDict()

        self.ldp = ldp

        self._processes = []
        self.p0 = None

    def __del__(self):
        if hasattr(self, u"_processes"):
            for p in self._processes:
                p.join()

    @property
    def nplanets(self):
        return self._nplanets

    def add_planet(self, r, a, e, T, phi, pomega, i):
        self._nplanets += 1
        self.rp = np.append(self.rp, r)
        self.ap = np.append(self.ap, a)
        self.ep = np.append(self.ep, e)
        self.tp = np.append(self.tp, T)
        self.php = np.append(self.php, phi)
        self.pop = np.append(self.pop, pomega)
        self.ip = np.append(self.ip, i)

    def to_vector(self):
        v = []

        for k, p in self._pars.iteritems():
            v.append(p.getter(self))

        return np.array(v, dtype=np.float64)

    def from_vector(self, v):
        ind, n = 0, self.nplanets

        for i, (k, p) in enumerate(self._pars.iteritems()):
            p.setter(self, v[i])

        return self

    def __call__(self, p):
        return self.lnprob(p)

    def lnprob(self, p):
        # Make sure that we catch all the over/under-flows.
        np.seterr(all=u"raise")
        try:
            self.from_vector(p)

            # Compute the prior.
            lnp = self.lnprior()
            if np.isinf(lnp) or np.isnan(lnp):
                return -np.inf

            # Compute the likelihood.
            lnl = self.lnlike()
            if np.isinf(lnl) or np.isnan(lnl):
                return -np.inf

        except FloatingPointError:
            return -np.inf

        return lnl + lnp

    def lnprior(self):
        lnp = 0.0

        # Priors on the limb darkening profile.
        ldp = self.ldp

        # LDP must be strictly positive.
        if np.any(ldp.intensity < 0) or np.any(ldp.intensity > 1):
            return -np.inf

        # The gammas in the quadratic case must sum to less than one and be
        # greater than or equal to zero.
        if hasattr(ldp, u"gamma1") and hasattr(ldp, u"gamma2"):
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

        # Add in the jitter.
        ivar = np.array(self._data[2])
        inds = ivar > 0
        ivar[inds] = 1. / (1. / ivar[inds] + self.jitter)

        chi2 = np.sum(delta * delta * ivar)

        # FIXME: regularization for jitter.

        return -0.5 * chi2

    def lightcurve(self, t=None):
        if t is None:
            assert self._data is not None
            t = self._data[0]
        return _bart.lightcurve(t, self.fs, self.iobs,
                                self.rp, self.ap, self.ep, self.tp, self.php,
                                self.pop, self.ip,
                                self.ldp.bins, self.ldp.intensity)

    def fit_for(self, *args):
        self._pars = OrderedDict()
        [self._fit_for(p) for p in args]

    def _fit_for(self, var):
        n = self.nplanets
        if var == u"fs":
            self._pars[u"fs"] = Parameter(r"$f_s$", u"fs")

        elif var in [u"T", u"r", u"a"]:
            if var == "T":
                tex, attr = r"$T_{0}$", u"tp"
            elif var == "r":
                tex, attr = r"$r_{0}$", u"rp"
            elif var == "a":
                tex, attr = r"$a_{0}$", u"ap"

            for i in range(n):
                self._pars[u"{0}{1}".format(var, i)] = LogParameter(
                    tex.format(i + 1), attr, i)

        elif var == u"e":
            tex, attr = r"$e_{0}$", u"ep"
            for i in range(n):
                self._pars[u"e{0}".format(i)] = Parameter(
                                tex.format(i + 1), attr=attr, ind=i)

        elif var in [u"phi", u"pomega"]:
            if var == u"phi":
                tex, attr = r"$\phi_{0}$", u"php"
            elif var == u"pomega":
                tex, attr = r"$\varpi_{0}$", u"pop"
            for i in range(n):
                self._pars[u"{0}{1}".format(var, i)] = ConstrainedParameter(
                    [0.0, 2 * np.pi], tex.format(i + 1), attr=attr, ind=i)

        elif var == u"i":
            tex, attr = r"$i_{0}$", u"ip"
            for i in range(n):
                self._pars[u"i{0}".format(i)] = Parameter(
                    tex.format(i + 1), attr=attr, ind=i)

        elif var == u"gamma":
            self._pars[u"gamma1"] = Parameter(r"$\gamma_1$", attr=u"ldp",
                                              ind=u"gamma1")
            self._pars[u"gamma2"] = Parameter(r"$\gamma_2$", attr=u"ldp",
                                              ind=u"gamma2")

        elif var == u"ldp":
            for i in range(len(self.ldp.intensity) - 1):
                self._pars[u"ldp_{0}".format(i)] = LDPParameter(
                        r"$\Delta I_{{{0}}}$".format(i), ind=i)

        elif var == u"jitter":
            self._pars[u"jitter"] = LogParameter(u"$s^2$", u"jitter")

        else:
            raise RuntimeError(u"Unknown parameter {0}".format(var))

    def _prepare_data(self, t, f, ferr):
        # Deal with masked and problematic data points.
        inds = ~(np.isnan(t) + np.isnan(f) + np.isnan(ferr)
            + np.isinf(t) + np.isinf(f) + np.isinf(ferr)
            + (t < 0) + (f < 0) + (ferr <= 0))
        t, f, ivar = t[inds], f[inds], 1.0 / ferr[inds] / ferr[inds]

        # Store the data.
        mu = np.median(f)
        self._data = [t - np.mean(t), f / mu, ivar * mu * mu]

    def optimize(self, t, f, ferr, pars=[u"fs", u"T", u"r", u"a", u"phi"]):
        self._prepare_data(t, f, ferr)
        self.fit_for(*pars)

        # Check vector conversions.
        p0 = self.to_vector()
        self.from_vector(p0)
        np.testing.assert_almost_equal(p0, self.to_vector())

        # Optimize.
        nll = lambda p: -self.lnprob(p)

        try:
            result = op.minimize(nll, p0)
        except FloatingPointError:
            print(u"Optimization failed. Returning last evaluated point.")
            return self.to_vector()

        if not result.success:
            print(u"Optimization was not successful.")

        self.from_vector(result.x)
        return result.x

    def fit(self, t=None, f=None, ferr=None,
            pars=[u"fs", u"T", u"r", u"a", u"phi"],
            threads=10, ntrim=2, nburn=300, niter=1000, thin=50,
            nwalkers=100,
            filename=u"./mcmc.h5", restart=None):

        if restart is not None:
            with h5py.File(restart, u"r") as f:
                self._data = tuple(f[u"data"])

                g = f[u"mcmc"]
                pars = g.attrs[u"pars"].split(u", ")
                threads = g.attrs[u"threads"]

                chain0 = g[u"chain"][...]
                lnp0 = g[u"lnp"][...]

                p0 = chain0[:, -1, :]
                nwalkers, i0, ndim = chain0.shape

            self.fit_for(*pars)
            s = emcee.EnsembleSampler(nwalkers, ndim, self, threads=threads)

        else:
            self._prepare_data(t, f, ferr)
            self.fit_for(*pars)
            p_init = self.to_vector()
            ndim = len(pars)

            size = 1e-6
            p0 = emcee.utils.sample_ball(p_init, size * p_init, size=nwalkers)
            i0 = 0

            s = emcee.EnsembleSampler(nwalkers, ndim, self, threads=threads)

            lp = s._get_lnprob(p0)[0]
            dlp = np.var(lp)
            while dlp > 2:
                size *= 0.5
                p0 = emcee.utils.sample_ball(p_init, size * p_init,
                                                size=nwalkers)

                lp = s._get_lnprob(p0)[0]
                dlp = np.var(lp)

        with h5py.File(filename, u"w") as f:
            f.create_dataset(u"data", data=np.vstack(self._data))

            g = f.create_group(u"mcmc")
            g.attrs[u"pars"] = u", ".join(pars)
            g.attrs[u"threads"] = threads
            g.attrs[u"ntrim"] = ntrim
            g.attrs[u"nburn"] = nburn
            g.attrs[u"niter"] = niter
            g.attrs[u"thin"] = thin

            N = i0 + int(niter / thin)
            c_ds = g.create_dataset(u"chain", (nwalkers, N, ndim),
                                    dtype=np.float64)
            lp_ds = g.create_dataset(u"lnp", (nwalkers, N),
                                    dtype=np.float64)

            if restart is not None:
                c_ds[:, :i0, :] = chain0
                lp_ds[:, :i0] = lnp0

        self._sampler = None

        if restart is None:
            for i in range(ntrim):
                p0, lprob, state = s.run_mcmc(p0, nburn, storechain=False)

                # Cluster to get rid of crap.
                mix = mog.MixtureModel(4, np.atleast_2d(lprob).T)
                mix.run_kmeans()
                rs, rmx = mix.kmeans_rs, np.argmax(mix.means.flatten())

                # Choose the "good" walkers.
                inds = rs == rmx
                good = p0[inds].T

                # Sample from the multi-dimensional Gaussian.
                mu, cov = np.mean(good, axis=1), np.cov(good)
                p0[~inds] = np.random.multivariate_normal(mu, cov,
                                                          np.sum(~inds))

                for n in np.arange(len(p0))[~inds]:
                    lp = self.lnprob(p0[n])
                    # NOTE: this _could_ go on forever.
                    while np.isinf(lp):
                        p0[n] = np.random.multivariate_normal(mu, cov)
                        lp = self.lnprob(p0[n])

                print(u"Rejected {0} walkers.".format(np.sum(~inds)))
                s.reset()

        # Reset and rerun.
        for i, (pos, lnprob, state) in enumerate(s.sample(p0,
                                                        thin=thin,
                                                        iterations=niter)):
            if i % thin == 0:
                print(np.mean(s.acceptance_fraction))
                with h5py.File(filename, u"a") as f:
                    g = f[u"mcmc"]
                    c_ds = g[u"chain"]
                    lp_ds = g[u"lnp"]

                    g.attrs[u"iterations"] = s.iterations
                    g.attrs[u"naccepted"] = s.naccepted
                    g.attrs[u"state"] = pickle.dumps(state)

                    ind = i0 + int(i / thin)
                    c_ds[:, ind, :] = pos
                    lp_ds[:, ind] = lnprob

        # Let's see some stats.
        print(u"Acceptance fraction: {0:.2f} %"
                .format(np.mean(s.acceptance_fraction)))

        self._sampler = s
        return self._sampler.flatchain

    def plot_fit(self, true_ldp=None):
        p = Process(target=_async_plot, args=(u"_lc_and_ldp", self, true_ldp))
        p.start()
        self._processes.append(p)

    def _lc_and_ldp(self, true_ldp):
        chain = self._sampler.flatchain

        # Compute the best fit period.
        if u"T0" in self._pars:
            T = np.exp(np.median(chain[:, self._pars.keys().index(u"T0")]))
        else:
            T = self.tp[0]

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
        pl.errorbar(time % T, flux, yerr=1.0 / np.sqrt(ivar), fmt=u".k")
        pl.plot(t, f, u"#4682b4", alpha=0.05)

        pl.savefig(u"lc.png")

        # Plot the limb-darkening.
        pl.clf()
        pl.plot(*ld, color=u"#4682b4", alpha=0.1)
        if true_ldp is not None:
            pl.plot(*true_ldp, color=u"k", lw=2)
        pl.savefig(u"ld.png")

    def plot_triangle(self, truths=None):
        p = Process(target=_async_plot, args=(u"_triangle", self, truths))
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

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        setattr(self, k, v)

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


class Prior(object):

    def __init__(self, *pars):
        self._pars = pars


class UniformPrior(Prior):

    def __call__(self, x):
        mn, mx = self._pars
        if mn <= x < mx:
            return 0.0
        return -np.inf

    def sample(self, N=1):
        mn, mx = self._pars
        return mn + (mx - mn) * np.random.rand(N)


class GaussianPrior(Prior):

    def __call__(self, x):
        mu, std = self._pars
        v = std * std
        return -0.5 * np.sum(((x - mu) / std) ** 2) \
               - 0.5 * np.log(2 * np.pi * v)

    def sample(self, N=1):
        mu, std = self._pars
        return mu + std * np.random.randn(N)


class Parameter(object):

    def __init__(self, name, attr=None, ind=None, prior=None):
        self.name = name
        self.attr = attr
        self.ind = ind
        self.prior = prior

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


class ConstrainedParameter(Parameter):

    def __init__(self, bounds, *args, **kwargs):
        super(ConstrainedParameter, self).__init__(*args, **kwargs)
        self.bounds = sorted(bounds)

    def setter(self, ps, val):
        v0 = self.iconv(val)
        while not self.bounds[0] < v0 < self.bounds[1]:
            v0 = self.bounds[0] + v0 % (self.bounds[1] - self.bounds[0])
        return super(ConstrainedParameter, self).setter(ps, self.conv(v0))


class LDPParameter(LogParameter):

    def getter(self, ps):
        return self.conv(ps.ldp.intensity[self.ind]
                         - ps.ldp.intensity[self.ind + 1])

    def setter(self, ps, val):
        j = self.ind
        ps.ldp.intensity.__setitem__(j + 1,
                                     ps.ldp.intensity[j] - self.iconv(val))
