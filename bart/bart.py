#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, unicode_literals

__all__ = ["Star", "Planet", "PlanetarySystem"]

from collections import OrderedDict
from multiprocessing import Process
import cPickle as pickle

import numpy as np
import scipy.optimize as op
import emcee
import triangle

try:
    import matplotlib.pyplot as pl
    pl = pl
except ImportError:
    pl = None

try:
    import h5py
    h5py = h5py
except ImportError:
    h5py = None

from bart import _bart, mog
from bart.ldp import LimbDarkening


_G = 2945.4625385377644


class Star(object):
    """
    Represents the parameters of a star that hosts a planetary system.

    :param mass: (optional)
        The mass of the star in Solar masses.

    :param radius: (optional)
        The radius of the star in Solar radii.

    :param flux: (optional)
        The flux of the star in arbitrary units.

    :param ldp: (optional)
        The limb-darkening profile (a subclass of :class:`LimbDarkening`) of
        the star. This will default to a uniformly illuminated star.

    """

    def __init__(self, mass=1.0, radius=1.0, flux=1.0, ldp=None):
        self.mass = mass
        self.radius = radius
        self.flux = flux

        # The limb darkening profile.
        if ldp is None:
            # Default to a uniformly illuminated star.
            self.ldp = LimbDarkening([1.0], [1.0])
        else:
            self.ldp = ldp

        # The list of fit parameters.
        self.parameters = []

    @property
    def vector(self):
        return np.concatenate([p.getter(self) for p in self.parameters])

    @vector.setter
    def vector(self, v):
        i = 0
        for p in self.parameters:
            p.setter(self, v[i:i + len(p)])

    def __len__(self):
        return np.sum([len(p) for p in self.parameters])


class Planet(object):
    """
    A :class:`Planet` object represents the set of parameters describing a
    single planet in a planetary system.

    :param r:
        The size of the planet in Solar radii.

    :param a:
        The semi-major axis of the orbit in Solar radii.

    :param t0: (optional)
        The time of a reference pericenter passage.

    :param e: (optional)
        The eccentricity of the orbit.

    :param pomega: (optional)
        The rotation of the orbital ellipse in the reference plane.

    :param ix: (optional)
        The inclination of the orbit around the perpendicular axis to the
        observer's line-of-sight in degrees.

    :param iy: (optional)
        The inclination of the orbit around the observes line-of-sight in
        degrees.

    """

    def __init__(self, r, a, t0=0.0, e=0.0, pomega=0.0, ix=0.0, iy=0.0):
        self.r = r
        self.a = a
        self.t0 = t0
        self.e = e
        self.pomega = pomega
        self.ix = ix
        self.iy = iy

    @property
    def vector(self):
        return np.array([self.r, self.a, self.t0, self.e, self.pomega,
                         self.ix, self.iy])

    def get_mstar(self, T):
        """
        Get the mass of the host star implied by the semi-major axis of this
        planet and an input period.

        """
        a = self.a
        return a * a * a * 4 * np.pi * np.pi / _G / T / T

    def get_period(self, mstar):
        """
        Get the period of this planet orbiting a star with a given mass.

        """
        a = self.a
        return 2 * np.pi * np.sqrt(a * a * a / _G / mstar)


class PlanetarySystem(object):
    """
    A system of planets orbiting a star.

    :param star:
        A :class:`Star` object.

    :param iobs: (optional)
        The viewing angle in degrees. 90 is defined as edge on.

    """

    def __init__(self, star, iobs=90.0):
        self._data = None

        # The properties of the system as a whole.
        self.iobs = iobs

        # The planets.
        self.planets = []

        # The fit parameters.
        self._pars = OrderedDict()

        # Process management book keeping.
        self._processes = []

    def __del__(self):
        if hasattr(self, u"_processes"):
            for p in self._processes:
                p.join()

    @property
    def nplanets(self):
        """
        The number of planets in the system.

        """
        return len(self.planets)

    def add_planet(self, planet):
        """
        Add a :class:`Planet` to the system.

        :param planet:
            The :class:`Planet` to add.

        """
        self.planets.append(planet)

    @property
    def vector(self):
        """
        Get the vector of fit parameters in the same order as they were added.

        """
        v = []
        for k, p in self._pars.iteritems():
            v.append(p.getter(self))
        return np.array(v, dtype=np.float64)

    @vector.setter
    def vector(self, v):
        """
        Set the current fit parameter values.

        """
        for i, (k, p) in enumerate(self._pars.iteritems()):
            p.setter(self, v[i])

    def __call__(self, p):
        return self.lnprob(p)

    def lnprob(self, p):
        """
        Compute the log-probability of the model evaluated at a given position.

        :param p:
            The vector of fit parameters where the model should be evaluated.

        """
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
        """
        Compute the log-prior of the current model.

        """
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

        if np.any(self.e < 0) or np.any(self.e > 1):
            return -np.inf

        return lnp

    def lnlike(self):
        """
        Compute the log-likelihood of the current model.

        """
        assert self._data is not None
        model = self.lightcurve(self._data[0])
        delta = self._data[1] - model

        # Add in the jitter.
        ivar = np.array(self._data[2])
        # inds = ivar > 0
        # ivar[inds] = 1. / (1. / ivar[inds] + self.jitter)

        chi2 = np.sum(delta * delta * ivar) - np.sum(np.log(ivar))
        return -0.5 * chi2

    def lightcurve(self, t):
        """
        Get the light curve of the model at the current model.

        :param t:
            The times where the light curve should be evaluated.

        """
        return _bart.lightcurve(t,
                                self.fstar, self.mstar, self.rstar, self.iobs,
                                self.r, self.a, self.e, self.t0, self.pomega,
                                self.incl,
                                self.ldp.bins, self.ldp.intensity)

    def fit_for(self, *args):
        self._pars = OrderedDict()
        [self._fit_for(p) for p in args]

    def _fit_for(self, var):
        n = self.nplanets
        if var == u"fstar":
            self._pars[u"fstar"] = Parameter(r"$f_\star$", u"fstar")

        elif var in [u"mstar", u"rstar"]:
            if var == "mstar":
                tex = r"$M_\star$"
            elif var == "rstar":
                tex = r"$R_\star$"

            self._pars[var] = LogParameter(tex, var)

        elif var in [u"t0", u"r", u"a"]:
            if var == "t0":
                tex = r"$t_0^{{({0})}}$"
            elif var == "r":
                tex = r"$r^{{({0})}}$"
            elif var == "a":
                tex = r"$a^{{({0})}}$"

            for i in range(n):
                self._pars[u"{0}{1}".format(var, i)] = LogParameter(
                    tex.format(i + 1), var, i)

        elif var == u"e":
            tex, attr = r"$e^{{({0})}}$", u"e"
            for i in range(n):
                self._pars[u"e{0}".format(i)] = Parameter(
                                tex.format(i + 1), attr=attr, ind=i)

        elif var == u"pomega":
            tex, attr = r"$\varpi^{{({0})}}$", u"pomoega"
            for i in range(n):
                self._pars[u"{0}{1}".format(var, i)] = ConstrainedParameter(
                    [0.0, 2 * np.pi], tex.format(i + 1), attr=attr, ind=i)

        elif var == u"i":
            tex, attr = r"$i^{{({0})}}$", u"incl"
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
        """
        Censor and prepare the data properly.

        """
        # Deal with masked and problematic data points.
        inds = ~(np.isnan(t) + np.isnan(f) + np.isnan(ferr)
            + np.isinf(t) + np.isinf(f) + np.isinf(ferr)
            + (f < 0) + (ferr <= 0))
        t, f, ivar = t[inds], f[inds], 1.0 / ferr[inds] / ferr[inds]

        # Store the data.
        mu = 1.0  # np.median(f)
        self._data = [t, f / mu, ivar * mu * mu]

    def optimize(self, t, f, ferr,
                 pars=[u"fstar", u"t0", u"r", u"a", u"pomega"]):
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

    def fit(self, data=None,
            pars=[u"fstar", u"t0", u"r", u"a", u"pomega"],
            threads=10, ntrim=2, nburn=300, niter=1000, thin=50,
            nwalkers=100,
            filename=u"./mcmc.h5", restart=None):
        """
        Fit the data using MCMC to get constraints on the parameters.

        :param data: (optional)
            A tuple of the form ``(t, f, ferr)`` giving the data to fit. This
            is required unless ``restart`` is set.

        """

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
            assert data is not None, "You need to provide some data to fit!"
            self._prepare_data(*data)
            self.fit_for(*pars)
            p_init = self.to_vector()
            ndim = len(p_init)

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
                print(u"Trimming pass {0}...".format(i + 1))
                p0, lprob, state = s.run_mcmc(p0, nburn, storechain=False)

                # Cluster to get rid of crap.
                mix = mog.MixtureModel(4, np.atleast_2d(lprob).T)
                mix.run_kmeans()
                rs, rmxs = mix.kmeans_rs, np.argsort(mix.means.flatten())

                # Choose the "good" walkers.
                for rmx in rmxs[::-1]:
                    inds = rs == rmx
                    good = p0[inds].T
                    if np.shape(good)[1] > 0:
                        break

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
                print(i, np.mean(s.acceptance_fraction))
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
                .format(100 * np.mean(s.acceptance_fraction)))

        try:
            print(u"Autocorrelation time: {0}".format(
                    thin * s.acor))
        except RuntimeError:
            print(u"Autocorrelation time: too short")

        self._sampler = s
        return self._sampler.flatchain

    def plot_fit(self, true_ldp=None):
        p = Process(target=_async_plot, args=(u"_lc_and_ldp", self, true_ldp))
        p.start()
        self._processes.append(p)

    def _lc_and_ldp(self, true_ldp):
        time, flux, ivar = self._data
        chain = self._sampler.flatchain

        # Compute the best fit period.
        for i in range(self.nplanets):
            if u"a{0}".format(i) in self._pars:
                a = np.exp(np.median(chain[:,
                                self._pars.keys().index(u"a{0}".format(i))]))
            else:
                a = self.a[i]

            T = self.planets[i].get_period(self.mstar)

            # Generate light curve samples.
            t = np.linspace(0, T, 500)
            t2 = np.linspace(time.min(), time.max(),
                    int(100 * (time.max() - time.min() / T)))
            f = np.empty((len(chain), len(t)))
            f2 = np.zeros(len(t2))
            ld = [self.ldp.plot()[0],
                  np.empty((len(chain), 2 * len(self.ldp.bins)))]
            for ind, v in enumerate(chain):
                f[ind, :] = self.from_vector(v).lightcurve(t)
                f2 += self.lightcurve(t2)
                ld[1][ind, :] = self.ldp.plot()[1]
            f = f.T
            f2 = f2.T / len(chain)
            ld[1] = ld[1].T

            # Plot the fit.
            mu = np.median(flux)
            pl.figure()
            ax = pl.axes([0.15, 0.15, 0.8, 0.8])
            ax.plot(time % T, 1000 * (flux / mu - 1), u".k", alpha=0.5)
            ax.set_xlabel(u"Phase [days]")
            ax.set_ylabel(r"Relative Brightness Variation [$\times 10^{-3}$]")
            pl.savefig(u"lc_{0}.png".format(i), dpi=300)

            ax.plot(t, 1000 * (f / mu - 1), u"#4682b4", alpha=0.04)
            pl.savefig(u"lc_fit_{0}.png".format(i), dpi=300)

            if i == 0:
                # Plot the full fit.
                pl.clf()
                ax = pl.axes([0.15, 0.15, 0.8, 0.8])
                ax.plot(time, 1000 * (flux / mu - 1), u".k", alpha=0.5)
                ax.set_xlabel(u"Time [days]")
                ax.set_ylabel(r"Relative Brightness Variation "
                              r"[$\times 10^{-3}$]")
                pl.savefig(u"lc_full.png", dpi=300)

                ax.plot(t2, 1000 * (f2 / mu - 1), u"#4682b4")
                pl.savefig(u"lc_full_fit.png", dpi=300)

        # Plot the limb-darkening.
        pl.clf()
        pl.plot(*ld, color=u"#4682b4", alpha=0.08)
        if true_ldp is not None:
            pl.plot(*true_ldp, color=u"k", lw=2)
        pl.savefig(u"ld.png", dpi=300)

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

        labels = [str(p) for i, (k, p) in enumerate(self._pars.iteritems())
                            if i in inds]

        # Add the log-prob values too.
        lp = self._sampler.lnprobability.flatten()
        lp /= np.max(lp)
        plotchain = np.hstack([plotchain,
                    np.atleast_2d(lp).T])
        inds.append(plotchain.shape[1] - 1)
        labels.append(r"$\propto$ log-prob")

        if truths is not None:
            truths = [truths.get(k)
                    for i, (k, p) in enumerate(self._pars.iteritems())
                    if i in inds]

        triangle.corner(plotchain[:, inds], labels=labels, bins=20,
                                truths=truths)

        pl.savefig(u"triangle.png")


def _async_plot(pltype, ps, *args):
    return getattr(ps, pltype)(*args)


class LDPParameter(LogParameter):

    def getter(self, ps):
        return self.conv(ps.ldp.intensity[self.ind]
                         - ps.ldp.intensity[self.ind + 1])

    def setter(self, ps, val):
        j = self.ind
        ps.ldp.intensity.__setitem__(j + 1,
                                     ps.ldp.intensity[j] - self.iconv(val))
